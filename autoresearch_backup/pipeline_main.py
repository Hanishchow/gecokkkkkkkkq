#!/usr/bin/env python3
"""
GEOCK Orchestration Pipeline
============================
Main pipeline orchestrator for data acquisition, model training, and evaluation.

USAGE:
  python pipeline_main.py --mode full        # Full pipeline
  python pipeline_main.py --mode acquire   # Data acquisition only
  python pipeline_main.py --mode train      # Training only
  python pipeline_main.py --mode evaluate   # Evaluation only
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
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


@dataclass
class PipelineRun:
    """Tracks a pipeline execution."""
    run_id: str
    mode: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    steps_completed: List[str] = None
    results: Dict = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.steps_completed is None:
            self.steps_completed = []
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []


class PipelineOrchestrator:
    """Main orchestrator for GEOCK pipeline."""
    
    def __init__(self, work_dir: Path = WORK_DIR, cache_dir: Path = CACHE_DIR):
        self.work_dir = work_dir
        self.cache_dir = cache_dir
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.acquire_script = work_dir / "pipeline_acquire.py"
        self.train_script = work_dir / "pipeline_train.py"
    
    def _run_script(self, script: Path, args: List[str]) -> Dict:
        """Run a pipeline script and return results."""
        cmd = [sys.executable, str(script)] + args
        log.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.stdout:
                log.info(result.stdout)
            
            if result.stderr:
                log.warning(result.stderr)
            
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {'status': 'completed' if result.returncode == 0 else 'failed'}
                
        except subprocess.TimeoutExpired:
            return {'status': 'timeout'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def get_status(self) -> Dict:
        """Get current pipeline status."""
        status = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'data': {},
            'models': {},
            'ready': False
        }
        
        existing_pdb = len(list((self.cache_dir / "lp_pdb_files").glob("*.pdb"))) if (self.cache_dir / "lp_pdb_files").exists() else 0
        
        training_paths = list(self.cache_dir.glob("lp_new_features_8k.pkl"))
        if training_paths:
            try:
                with open(training_paths[0], 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, list):
                    status['data']['training_records'] = len(data)
            except:
                pass
        
        lp_csv = self.cache_dir / "LP_PDBBind.csv"
        if lp_csv.exists():
            import pandas as pd
            df = pd.read_csv(lp_csv)
            status['data']['lp_pdbbind_records'] = len(df)
            status['data']['pdb_downloaded'] = existing_pdb
            status['data']['pdb_missing'] = len(df) - existing_pdb
        
        model_files = list(self.work_dir.glob("geock_*_latest.pkl"))
        for mf in model_files:
            try:
                with open(mf, 'rb') as f:
                    m = pickle.load(f)
                model_name = mf.stem.replace('geock_', '').replace('_latest', '')
                status['models'][model_name] = {
                    'test_r': m.get('test_r', 'N/A'),
                    'cv_r': m.get('cv_r', 'N/A'),
                    'file': str(mf)
                }
            except:
                pass
        
        status['ready'] = (
            status['data'].get('training_records', 0) > 1000 and
            len(status['models']) > 0
        )
        
        return status
    
    def run_acquire(self, limit: int = None) -> Dict:
        """Run data acquisition pipeline."""
        log.info("=" * 60)
        log.info("PHASE: Data Acquisition")
        log.info("=" * 60)
        
        results = {}
        
        result = self._run_script(self.acquire_script, ['--step', 'fetch_pdb'])
        results['fetch_pdb'] = result
        
        result = self._run_script(self.acquire_script, ['--step', 'extract_features'])
        results['extract_features'] = result
        
        result = self._run_script(self.acquire_script, ['--step', 'fetch_chembl'])
        results['fetch_chembl'] = result
        
        result = self._run_script(self.acquire_script, ['--step', 'combine_data'])
        results['combine_data'] = result
        
        return results
    
    def run_train(self) -> Dict:
        """Run model training pipeline."""
        log.info("=" * 60)
        log.info("PHASE: Model Training")
        log.info("=" * 60)
        
        result = self._run_script(self.train_script, ['--model', 'compare'])
        return result
    
    def run_full(self) -> Dict:
        """Run full pipeline."""
        log.info("=" * 60)
        log.info(f"GEOCK Full Pipeline - Run {self.run_id}")
        log.info("=" * 60)
        
        results = {
            'run_id': self.run_id,
            'started_at': datetime.now().isoformat(),
            'steps': {}
        }
        
        results['steps']['acquire'] = self.run_acquire()
        results['steps']['train'] = self.run_train()
        results['steps']['status'] = self.get_status()
        
        results['completed_at'] = datetime.now().isoformat()
        
        report_path = self.work_dir / f"pipeline_run_{self.run_id}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        log.info(f"Pipeline run complete. Report: {report_path}")
        
        return results
    
    def run_evaluate(self) -> Dict:
        """Run model evaluation."""
        log.info("=" * 60)
        log.info("PHASE: Model Evaluation")
        log.info("=" * 60)
        
        status = self.get_status()
        
        eval_results = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'data_summary': status['data'],
            'models': status['models'],
            'pipeline_ready': status['ready']
        }
        
        eval_path = self.work_dir / f"evaluation_{self.run_id}.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        log.info(f"Evaluation complete: {eval_path}")
        
        return eval_results


def main():
    parser = argparse.ArgumentParser(description="GEOCK Pipeline Orchestrator")
    parser.add_argument('--mode', type=str, default='status',
                        choices=['full', 'acquire', 'train', 'evaluate', 'status'],
                        help='Pipeline mode')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit records for testing')
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    if args.mode == 'status':
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
    elif args.mode == 'acquire':
        results = orchestrator.run_acquire(limit=args.limit)
        print(json.dumps(results, indent=2))
    elif args.mode == 'train':
        results = orchestrator.run_train()
        print(json.dumps(results, indent=2))
    elif args.mode == 'evaluate':
        results = orchestrator.run_evaluate()
        print(json.dumps(results, indent=2))
    elif args.mode == 'full':
        results = orchestrator.run_full()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
