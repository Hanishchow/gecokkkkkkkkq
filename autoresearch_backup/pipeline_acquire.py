#!/usr/bin/env python3
"""
GEOCK Data Acquisition Pipeline
================================
Comprehensive pipeline for extracting PDB files and training data.

SOURCES:
  1. LP-PDBBind (GitHub: THGLab/LP-PDBBind) - 19,443 binding records
  2. ChEMBL (via API) - Drug-like compounds
  3. PDBBind Core (manual download) - 285 high-quality complexes

USAGE:
  python pipeline_acquire.py --step fetch_pdb      # Download PDB files
  python pipeline_acquire.py --step extract_features  # Generate features
  python pipeline_acquire.py --step combine_data    # Combine all data
  python pipeline_acquire.py --step all           # Run all steps
"""

import os
import sys
import json
import pickle
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
DATA_DIR = Path("/mnt/c/Users/yakka/Downloads")
WORK_DIR = Path("/home/chow/autoresearch")
BATCH_SIZE = 100
MAX_WORKERS = 16
RCSB_API = "https://data.rcsb.org/rest"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


@dataclass
class PipelineState:
    """Tracks pipeline execution state for resumability."""
    step: str = ""
    started_at: str = ""
    completed_at: str = ""
    records_processed: int = 0
    records_failed: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'PipelineState':
        if path.exists():
            with open(path) as f:
                return cls(**json.load(f))
        return cls()


class PDBFetcher:
    """Fetch PDB files from RCSB."""
    
    def __init__(self, cache_dir: Path, max_workers: int = MAX_WORKERS):
        self.cache_dir = cache_dir
        self.pdb_dir = cache_dir / "lp_pdb_files"
        self.pdb_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GEOCK-Pipeline/1.0'})
    
    def get_existing(self) -> set:
        """Get set of already-downloaded PDB IDs."""
        if not self.pdb_dir.exists():
            return set()
        return {f.stem.upper() for f in self.pdb_dir.glob("*.pdb")}
    
    def fetch_pdb(self, pdb_id: str) -> Tuple[str, bool, str]:
        """Download a single PDB file."""
        pdb_id = pdb_id.upper().strip()
        if len(pdb_id) != 4:
            return pdb_id, False, "Invalid PDB ID"
        
        output_path = self.pdb_dir / f"{pdb_id.lower()}.pdb"
        if output_path.exists():
            return pdb_id, True, "Already exists"
        
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200 and "HEADER" in resp.text:
                with open(output_path, 'w') as f:
                    f.write(resp.text)
                return pdb_id, True, "Success"
            return pdb_id, False, f"HTTP {resp.status_code}"
        except Exception as e:
            return pdb_id, False, str(e)
    
    def fetch_batch(self, pdb_ids: List[str], batch_name: str = "") -> Dict:
        """Fetch multiple PDB files in parallel."""
        existing = self.get_existing()
        to_fetch = [p for p in pdb_ids if p.upper() not in existing]
        
        if not to_fetch:
            log.info(f"{batch_name}: All {len(pdb_ids)} already downloaded")
            return {"downloaded": 0, "failed": 0, "skipped": len(pdb_ids)}
        
        results = {"downloaded": 0, "failed": 0, "skipped": existing}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(self.fetch_pdb, pid): pid for pid in to_fetch}
            
            for i, future in enumerate(as_completed(futures), 1):
                pdb_id, ok, msg = future.result()
                if ok:
                    results["downloaded"] += 1
                else:
                    results["failed"] += 1
                    results.setdefault("errors", []).append(f"{pdb_id}: {msg}")
                
                if i % 100 == 0:
                    log.info(f"  {batch_name}: {i}/{len(to_fetch)} - OK:{results['downloaded']} FAIL:{results['failed']}")
        
        log.info(f"{batch_name}: Downloaded={results['downloaded']}, Failed={results['failed']}")
        return results
    
    def get_lp_pdbbind(self) -> pd.DataFrame:
        """Load LP-PDBBind CSV."""
        csv_path = self.cache_dir / "LP_PDBBind.csv"
        if not csv_path.exists():
            csv_path = DATA_DIR / "LP_PDBBind.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"LP-PDBBind CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        log.info(f"LP-PDBBind: {len(df)} records")
        return df


class ChEMBLFetcher:
    """Fetch data from ChEMBL API."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'GEOCK-Pipeline/1.0'})
        self.session.params = {'format': 'json'}
    
    def fetch_page(self, endpoint: str, page: int = 1, page_size: int = 100) -> Optional[dict]:
        """Fetch a page from ChEMBL API."""
        url = f"{CHEMBL_API}/{endpoint}"
        params = {'page': page, 'limit': page_size}
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            log.warning(f"ChEMBL API error: {e}")
        return None
    
    def fetch_assays(self, target_chembl_id: str, min_activity: float = 5.0) -> List[Dict]:
        """Fetch activity data for a target."""
        activities = []
        page = 1
        
        while True:
            data = self.fetch_page(
                f"activity?target_chembl_id={target_chembl_id}",
                page=page
            )
            if not data or 'activities' not in data:
                break
            
            for act in data['activities']:
                try:
                    smiles = act.get('canonical_smiles')
                    affinity = act.get('pchembl_value') or act.get('p_activity_value')
                    
                    if smiles and affinity and float(affinity) >= min_activity:
                        activities.append({
                            'smiles': smiles,
                            'pKd': float(affinity),
                            'chembl_id': act.get('molecule_chembl_id'),
                            'assay_id': act.get('assay_chembl_id'),
                            'target': target_chembl_id
                        })
                except (ValueError, TypeError):
                    continue
            
            if not data.get('page_meta', {}).get('next'):
                break
            page += 1
        
        return activities


class FeatureExtractor:
    """Extract features from molecules and PDB files."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    
    def smiles_to_features(self, smiles: str) -> Optional[Dict]:
        """Extract features from SMILES."""
        if not smiles or pd.isna(smiles):
            return None
        try:
            smiles = str(smiles).strip()
            if len(smiles) < 3:
                return None
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            ecfp_gen = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=512
            )
            ecfp = np.array(ecfp_gen, dtype=np.float32)
            
            features = {
                'smiles': smiles,
                'ecfp': ecfp,
                'mol_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_h_acceptors': Lipinski.NumHAcceptors(mol),
                'num_h_donors': Lipinski.NumHDonors(mol),
                'num_rotatable': Lipinski.NumRotatableBonds(mol),
                'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
                'num_rings': Lipinski.RingCount(mol),
            }
            return features
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}")
            return None
    
    def pdb_to_features(self, pdb_path: Path) -> Optional[Dict]:
        """Extract features from PDB file."""
        try:
            with open(pdb_path) as f:
                pdb_content = f.read()
            
            lines = pdb_content.split('\n')
            
            features = {
                'pdb_id': pdb_path.stem,
                'num_atoms': sum(1 for l in lines if l.startswith('ATOM')),
                'num_residues': len(set(
                    f"{l[17:20]}{l[22:26].strip()}" 
                    for l in lines if l.startswith('ATOM') and len(l) >= 27
                )),
                'num_ligand_atoms': 0,
                'has_ligand': False,
            }
            
            for l in lines:
                if l.startswith('HETATM') and l[17:20].strip() not in ['HOH', 'MSE', 'UNK']:
                    features['has_ligand'] = True
                    features['num_ligand_atoms'] += 1
            
            return features
        except Exception as e:
            log.warning(f"PDB parsing failed: {e}")
            return None


class DataCombiner:
    """Combine data from multiple sources."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    
    def load_existing(self) -> List[Dict]:
        """Load existing training data."""
        paths = [
            self.cache_dir / "lp_new_features_8k.pkl",
            self.cache_dir / "lp_all_features.pkl",
            self.cache_dir / "chembl_more.pkl",
        ]
        
        all_data = []
        for path in paths:
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    elif isinstance(data, dict):
                        all_data.append(data)
                    log.info(f"Loaded {len(data)} records from {path.name}")
                except Exception as e:
                    log.warning(f"Failed to load {path}: {e}")
        
        return all_data
    
    def deduplicate(self, data: List[Dict]) -> List[Dict]:
        """Remove duplicate SMILES."""
        seen = set()
        unique = []
        
        for record in data:
            smiles = record.get('smiles', '')
            if smiles and smiles not in seen:
                seen.add(smiles)
                unique.append(record)
        
        log.info(f"Deduplication: {len(data)} → {len(unique)}")
        return unique
    
    def validate_record(self, record: Dict) -> bool:
        """Validate a training record."""
        required = ['smiles', 'ecfp', 'affinity']
        return all(k in record and record[k] is not None for k in required)


class Pipeline:
    """Main data acquisition pipeline orchestrator."""
    
    def __init__(self, work_dir: Path = WORK_DIR, cache_dir: Path = CACHE_DIR):
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.state_file = work_dir / "pipeline_state.json"
        
        self.pdb_fetcher = PDBFetcher(cache_dir)
        self.chembl_fetcher = ChEMBLFetcher(cache_dir)
        self.feature_extractor = FeatureExtractor(cache_dir)
        self.data_combiner = DataCombiner(cache_dir)
        
        self.state = PipelineState.load(self.state_file)
    
    def step_fetch_pdb(self, limit: int = None) -> Dict:
        """Step 1: Fetch PDB files from LP-PDBBind."""
        log.info("=" * 60)
        log.info("STEP: Fetch PDB files")
        log.info("=" * 60)
        
        state = PipelineState(step="fetch_pdb", started_at=datetime.now().isoformat())
        
        try:
            df = self.pdb_fetcher.get_lp_pdbbind()
            pdb_ids = df['Unnamed: 0'].dropna().unique().tolist()
            
            if limit:
                pdb_ids = pdb_ids[:limit]
            
            existing = self.pdb_fetcher.get_existing()
            to_fetch = [p for p in pdb_ids if p.upper() not in existing]
            
            log.info(f"Total PDB IDs: {len(pdb_ids)}")
            log.info(f"Already downloaded: {len(existing)}")
            log.info(f"Need to fetch: {len(to_fetch)}")
            
            batch_size = 500
            total_downloaded = 0
            total_failed = 0
            
            for i in range(0, len(to_fetch), batch_size):
                batch = to_fetch[i:i+batch_size]
                results = self.pdb_fetcher.fetch_batch(batch, f"Batch-{i//batch_size + 1}")
                total_downloaded += results['downloaded']
                total_failed += results['failed']
                
                state.records_processed += results['downloaded']
                state.records_failed += results['failed']
            
            state.completed_at = datetime.now().isoformat()
            state.metadata = {
                'total_pdb_ids': len(pdb_ids),
                'existing': len(existing),
                'downloaded': total_downloaded,
                'failed': total_failed
            }
            
            log.info(f"PDB Fetch Complete: {total_downloaded} downloaded, {total_failed} failed")
            
        except Exception as e:
            log.error(f"PDB fetch failed: {e}")
            state.errors.append(str(e))
        
        state.save(self.state_file)
        return asdict(state)
    
    def step_extract_features(self, limit: int = None) -> Dict:
        """Step 2: Extract features from downloaded PDBs."""
        log.info("=" * 60)
        log.info("STEP: Extract Features")
        log.info("=" * 60)
        
        state = PipelineState(step="extract_features", started_at=datetime.now().isoformat())
        
        try:
            df = self.pdb_fetcher.get_lp_pdbbind()
            pdb_dir = self.pdb_fetcher.pdb_dir
            
            existing_features = self.cache_dir / "lp_new_features_8k.pkl"
            processed_ids = set()
            
            if existing_features.exists():
                with open(existing_features, 'rb') as f:
                    existing = pickle.load(f)
                processed_ids = {c.get('pdb_id') for c in existing if 'pdb_id' in c}
                log.info(f"Already processed: {len(processed_ids)} PDBs")
            
            features_list = list(existing) if existing_features.exists() else []
            new_count = 0
            
            for _, row in df.iterrows():
                pdb_id = row.get('Unnamed: 0')  # LP-PDBBind uses 'Unnamed: 0' for PDB ID
                if not pdb_id or pdb_id in processed_ids:
                    continue
                
                pdb_path = pdb_dir / f"{pdb_id.lower()}.pdb"
                if not pdb_path.exists():
                    continue
                
                pdb_features = self.feature_extractor.pdb_to_features(pdb_path)
                smiles = row.get('smiles')
                affinity = row.get('value')  # LP-PDBBind uses 'value' for affinity
                
                if smiles and affinity:
                    mol_features = self.feature_extractor.smiles_to_features(smiles)
                    if mol_features:
                        features_list.append({
                            **mol_features,
                            **pdb_features,
                            'affinity': float(affinity),
                            'pdb_id': pdb_id,
                            'source': 'lp_pdbbind'
                        })
                        new_count += 1
                        processed_ids.add(pdb_id)
                        
                        if new_count % 1000 == 0:
                            log.info(f"  Processed: {new_count} new records")
                
                if limit and new_count >= limit:
                    break
            
            output_path = self.cache_dir / "lp_new_features_8k.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(features_list, f)
            
            state.records_processed = len(features_list)
            state.metadata = {'new_records': new_count, 'total': len(features_list)}
            state.completed_at = datetime.now().isoformat()
            
            log.info(f"Feature Extraction Complete: {new_count} new, {len(features_list)} total")
            
        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            state.errors.append(str(e))
        
        state.save(self.state_file)
        return asdict(state)
    
    def step_fetch_chembl(self, targets: List[str] = None, limit: int = 10000) -> Dict:
        """Step 3: Fetch additional data from ChEMBL."""
        log.info("=" * 60)
        log.info("STEP: Fetch ChEMBL Data")
        log.info("=" * 60)
        
        state = PipelineState(step="fetch_chembl", started_at=datetime.now().isoformat())
        
        if targets is None:
            targets = [
                'CHEMBL1937',  # HIV protease
                'CHEMBL3409',  # Cyclin-dependent kinase 2
                'CHEMBL4182',  # EGFR
                'CHEMBL1825',  # COX-2
            ]
        
        all_activities = []
        
        for target in targets:
            log.info(f"Fetching ChEMBL target: {target}")
            activities = self.chembl_fetcher.fetch_assays(target)
            all_activities.extend(activities)
            log.info(f"  Got {len(activities)} activities")
            
            if len(all_activities) >= limit:
                break
        
        features_list = []
        for act in all_activities[:limit]:
            feat = self.feature_extractor.smiles_to_features(act['smiles'])
            if feat:
                features_list.append({
                    **feat,
                    'affinity': act['pKd'],
                    'source': 'chembl',
                    'target': act['target']
                })
        
        output_path = self.cache_dir / f"chembl_{datetime.now().strftime('%Y%m%d')}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(features_list, f)
        
        state.records_processed = len(features_list)
        state.metadata = {'source': 'chembl', 'targets': targets}
        state.completed_at = datetime.now().isoformat()
        
        log.info(f"ChEMBL Fetch Complete: {len(features_list)} records")
        
        state.save(self.state_file)
        return asdict(state)
    
    def step_combine_data(self) -> Dict:
        """Step 4: Combine all data sources."""
        log.info("=" * 60)
        log.info("STEP: Combine Data")
        log.info("=" * 60)
        
        state = PipelineState(step="combine_data", started_at=datetime.now().isoformat())
        
        try:
            all_data = self.data_combiner.load_existing()
            log.info(f"Loaded {len(all_data)} total records")
            
            unique_data = self.data_combiner.deduplicate(all_data)
            valid_data = [d for d in unique_data if self.data_combiner.validate_record(d)]
            log.info(f"Valid records: {len(valid_data)}")
            
            output_path = self.cache_dir / "geock_training_data.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(valid_data, f)
            
            output_stats = {
                'total_records': len(valid_data),
                'sources': {},
                'affinity_stats': {
                    'mean': float(np.mean([d['affinity'] for d in valid_data])),
                    'std': float(np.std([d['affinity'] for d in valid_data])),
                    'min': float(np.min([d['affinity'] for d in valid_data])),
                    'max': float(np.max([d['affinity'] for d in valid_data])),
                }
            }
            
            for d in valid_data:
                src = d.get('source', 'unknown')
                output_stats['sources'][src] = output_stats['sources'].get(src, 0) + 1
            
            stats_path = self.cache_dir / "geock_training_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(output_stats, f, indent=2)
            
            state.records_processed = len(valid_data)
            state.metadata = output_stats
            state.completed_at = datetime.now().isoformat()
            
            log.info(f"Data Combination Complete: {len(valid_data)} records")
            log.info(f"Stats: {output_stats['sources']}")
            
        except Exception as e:
            log.error(f"Data combination failed: {e}")
            state.errors.append(str(e))
        
        state.save(self.state_file)
        return asdict(state)
    
    def step_generate_report(self) -> Dict:
        """Generate pipeline execution report."""
        log.info("=" * 60)
        log.info("STEP: Generate Report")
        log.info("=" * 60)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "pipeline_state": asdict(self.state),
            "data_summary": {},
            "next_steps": []
        }
        
        try:
            df = self.pdb_fetcher.get_lp_pdbbind()
            existing_pdb = self.pdb_fetcher.get_existing()
            
            report["data_summary"] = {
                "lp_pdbbind_records": len(df),
                "downloaded_pdb_files": len(existing_pdb),
                "missing_pdb_files": len(df) - len(existing_pdb),
                "download_progress": f"{len(existing_pdb) / len(df) * 100:.1f}%"
            }
            
            training_path = self.cache_dir / "geock_training_data.pkl"
            if training_path.exists():
                with open(training_path, 'rb') as f:
                    training_data = pickle.load(f)
                report["data_summary"]["training_records"] = len(training_data)
            
            if report["data_summary"].get("missing_pdb_files", 0) > 0:
                report["next_steps"].append("Continue PDB download")
            if report["data_summary"].get("training_records", 0) < 10000:
                report["next_steps"].append("Add more training data sources")
            
        except Exception as e:
            log.error(f"Report generation failed: {e}")
        
        report_path = self.work_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        log.info(f"Report saved: {report_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="GEOCK Data Acquisition Pipeline")
    parser.add_argument('--step', type=str, required=True,
                        choices=['fetch_pdb', 'extract_features', 'fetch_chembl', 'combine_data', 'all'],
                        help='Pipeline step to execute')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of records to process')
    parser.add_argument('--targets', type=str, nargs='+',
                        help='ChEMBL target IDs')
    
    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    if args.step == 'fetch_pdb':
        result = pipeline.step_fetch_pdb(limit=args.limit)
    elif args.step == 'extract_features':
        result = pipeline.step_extract_features(limit=args.limit)
    elif args.step == 'fetch_chembl':
        result = pipeline.step_fetch_chembl(targets=args.targets, limit=args.limit or 10000)
    elif args.step == 'combine_data':
        result = pipeline.step_combine_data()
    elif args.step == 'all':
        log.info("Running full pipeline...")
        pipeline.step_fetch_pdb(limit=args.limit)
        pipeline.step_extract_features()
        pipeline.step_fetch_chembl(limit=args.limit or 10000)
        pipeline.step_combine_data()
        result = pipeline.step_generate_report()
    
    if result:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
