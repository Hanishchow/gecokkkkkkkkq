"""
Vina-Style Physics Engine for GEOCK
Implements: BFGS optimization, Metropolis criterion, RMSD clustering, Simulated annealing

Based on AutoDock Vina source code principles (without using Vina itself)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Pose:
    """Represents a molecular pose"""
    coordinates: np.ndarray  # [N_atoms, 3]
    torsions: np.ndarray   # [N_torsions]
    energy: float = 0.0
    
    def copy(self):
        return Pose(
            coordinates=self.coordinates.copy(),
            torsions=self.torsions.copy(),
            energy=self.energy
        )


class VinaStyleOptimizer:
    """
    Implements Vina-style optimization:
    - BFGS quasi-Newton local optimization
    - Metropolis criterion with simulated annealing
    - RMSD-based pose clustering
    """
    
    def __init__(self, scoring_function, n_atoms=50, n_torsions=10):
        self.scoring_function = scoring_function
        self.n_atoms = n_atoms
        self.n_torsions = n_torsions
        
    def docking_search(
        self,
        initial_pose: Pose,
        exhaustiveness: int = 8,
        max_steps: int = 100,
        temperature_init: float = 100.0,
        temperature_final: float = 0.0,
    ) -> List[Pose]:
        """
        Main Vina-style docking search
        Multiple independent runs with mutation + BFGS + Metropolis
        """
        
        all_poses = []
        
        for run in range(exhaustiveness):
            # Generate random starting conformation
            start_pose = self.mutate_conformation(
                initial_pose,
                translation_magnitude=2.0,
                rotation_magnitude=0.5,
                torsion_magnitude=np.pi / 4
            )
            
            # Run optimization
            poses = self.run_single_search(
                start_pose,
                max_steps=max_steps,
                temp_init=temperature_init,
                temp_final=temperature_final
            )
            
            all_poses.extend(poses)
        
        return all_poses
    
    def run_single_search(
        self,
        initial_pose: Pose,
        max_steps: int = 100,
        temp_init: float = 100.0,
        temp_final: float = 0.0,
    ) -> List[Pose]:
        """
        Single search run: mutation + BFGS + Metropolis acceptance
        """
        
        current = initial_pose.copy()
        current.energy = self.scoring_function(current)
        
        best = current.copy()
        results = []
        
        for step in range(max_steps):
            # Calculate temperature (simulated annealing)
            progress = step / max_steps
            temperature = temp_init * (1.0 - progress) + temp_final * progress
            
            # Random perturbation
            perturbed = self.mutate_conformation(
                current,
                translation_magnitude=0.2 * (1.0 - progress),
                rotation_magnitude=0.1 * (1.0 - progress),
                torsion_magnitude=0.1 * (1.0 - progress)
            )
            
            # BFGS local optimization
            optimized, energy = self.bfgs_optimization(
                perturbed,
                max_iterations=50
            )
            
            # Metropolis acceptance
            if self.metropolis_accept(current.energy, energy, temperature):
                current = optimized
                current.energy = energy
            
            # Remember best
            if energy < best.energy:
                best = optimized.copy()
                best.energy = energy
                results.append(best.copy())
        
        return results
    
    def mutate_conformation(
        self,
        pose: Pose,
        translation_magnitude: float = 0.2,
        rotation_magnitude: float = 0.1,
        torsion_magnitude: float = 0.1,
    ) -> Pose:
        """
        Random mutation: translation + rotation + torsion changes
        """
        
        mutated = pose.copy()
        
        # Random translation (brownian motion)
        delta_pos = np.random.normal(0, translation_magnitude, size=(self.n_atoms, 3))
        mutated.coordinates += delta_pos
        
        # Random torsion changes
        if self.n_torsions > 0 and len(pose.torsions) > 0:
            n_to_change = min(self.n_torsions, len(pose.torsions))
            indices = np.random.choice(len(pose.torsions), n_to_change, replace=False)
            for i in indices:
                mutated.torsions[i] += np.random.normal(0, torsion_magnitude)
                mutated.torsions[i] = mutated.torsions[i] % (2 * np.pi)
        
        return mutated
    
    def metropolis_accept(self, energy_old: float, energy_new: float, temperature: float) -> bool:
        """
        Metropolis criterion: accept worse solutions probabilistically
        P(accept) = 1 if ΔE < 0 (better)
        P(accept) = exp(-ΔE/kT) if ΔE ≥ 0 (worse)
        """
        
        delta_energy = energy_new - energy_old
        
        if delta_energy < 0:
            return True  # Always accept better
        
        # Boltzmann factor
        k_B = 1.987e-3  # kcal/(mol·K)
        boltz = np.exp(-delta_energy / (k_B * max(temperature, 0.001)))
        
        return np.random.random() < boltz
    
    def bfgs_optimization(
        self,
        initial_pose: Pose,
        max_iterations: int = 300,
        tolerance: float = 1e-5,
    ) -> Tuple[Pose, float]:
        """
        BFGS quasi-Newton optimization
        More efficient than gradient descent
        """
        
        # Flatten to vector
        x = self.pose_to_vector(initial_pose)
        n = len(x)
        
        # Initialize Hessian approximation
        H = np.eye(n)
        
        # Initial evaluation
        current_pose = self.vector_to_pose(x, initial_pose)
        current_energy = self.scoring_function(current_pose)
        current_grad = self.compute_gradient(x)
        
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            # Search direction: p = -H * g
            p = -np.dot(H, current_grad)
            
            # Ensure downhill
            if np.dot(current_grad, p) >= 0:
                H = np.eye(n)
                p = -current_grad
            
            # Line search
            alpha = self.line_search(x, p, current_energy, current_grad)
            
            # Update
            x_new = x + alpha * p
            new_pose = self.vector_to_pose(x_new, initial_pose)
            new_energy = self.scoring_function(new_pose)
            new_grad = self.compute_gradient(x_new)
            
            # BFGS update
            s = x_new - x
            y = new_grad - current_grad
            
            s_dot_y = np.dot(s, y)
            if abs(s_dot_y) > 1e-10:
                # First term
                yy = np.outer(y, y) / s_dot_y
                # Second term
                Hs = np.dot(H, s)
                s_Hs = np.dot(s, Hs)
                if abs(s_Hs) > 1e-10:
                    HssH = np.outer(Hs, Hs) / s_Hs
                    H = H + yy - HssH
            
            # Check convergence
            grad_norm = np.linalg.norm(new_grad)
            energy_change = abs(new_energy - current_energy)
            
            if grad_norm < tolerance and energy_change < tolerance:
                converged = True
            
            x = x_new
            current_energy = new_energy
            current_grad = new_grad
            iteration += 1
        
        final_pose = self.vector_to_pose(x, initial_pose)
        return final_pose, current_energy
    
    def line_search(
        self,
        x: np.ndarray,
        p: np.ndarray,
        f0: float,
        g0: np.ndarray,
        initial_alpha: float = 0.1,
        c: float = 1e-4,
    ) -> float:
        """
        Backtracking line search (Armijo condition)
        """
        
        alpha = initial_alpha
        
        for _ in range(20):
            x_new = x + alpha * p
            new_pose = self.vector_to_pose(x_new, self.vector_to_pose(x, None))
            f_new = self.scoring_function(new_pose)
            
            # Armijo condition
            if f_new <= f0 + c * alpha * np.dot(g0, p):
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def pose_to_vector(self, pose: Pose) -> np.ndarray:
        """Flatten pose to 1D vector"""
        return pose.coordinates.flatten()
    
    def vector_to_pose(self, x: np.ndarray, template: Pose) -> Pose:
        """Restore pose from vector"""
        n_atoms = len(template.coordinates)
        coords = x[:n_atoms*3].reshape(n_atoms, 3)
        
        n_tors = len(template.torsions) if template.torsions is not None else 0
        torsions = x[n_atoms*3:] if n_tors > 0 else np.array([])
        
        return Pose(coordinates=coords, torsions=torsions)
    
    def compute_gradient(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        """
        Numerical gradient computation
        """
        n = len(x)
        grad = np.zeros(n)
        
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            
            pose_plus = self.vector_to_pose(x_plus, self.vector_to_pose(x, None))
            pose_minus = self.vector_to_pose(x - h * np.eye(n)[i], self.vector_to_pose(x, None))
            
            f_plus = self.scoring_function(pose_plus)
            f_minus = self.scoring_function(pose_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * h)
        
        return grad


class PoseClustering:
    """
    Vina-style RMSD-based pose clustering
    """
    
    @staticmethod
    def cluster_poses(poses: List[Pose], rmsd_threshold: float = 2.0) -> List[List[Pose]]:
        """
        Cluster similar poses within RMSD threshold
        Keep lowest energy pose per cluster
        """
        
        if not poses:
            return []
        
        # Sort by energy
        sorted_poses = sorted(poses, key=lambda p: p.energy)
        
        clusters = []
        
        for pose in sorted_poses:
            assigned = False
            
            for cluster in clusters:
                # RMSD to cluster representative
                rmsd = PoseClustering.calculate_rmsd(
                    pose.coordinates,
                    cluster[0].coordinates
                )
                
                if rmsd < rmsd_threshold:
                    cluster.append(pose)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([pose])
        
        return clusters
    
    @staticmethod
    def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
        """
        Calculate RMSD between two sets of coordinates
        """
        
        if coords1.shape != coords2.shape:
            min_len = min(len(coords1), len(coords2))
            coords1 = coords1[:min_len]
            coords2 = coords2[:min_len]
        
        return np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))
    
    @staticmethod
    def rank_clusters(clusters: List[List[Pose]]) -> List[Dict]:
        """
        Rank clusters by lowest energy pose
        """
        
        results = []
        
        for i, cluster in enumerate(clusters):
            best_pose = min(cluster, key=lambda p: p.energy)
            
            # Calculate RMSD bounds
            rmsd_lb = 0.0
            rmsd_ub = max(
                PoseClustering.calculate_rmsd(p.coordinates, best_pose.coordinates)
                for p in cluster
            )
            
            results.append({
                'mode': i + 1,
                'affinity': best_pose.energy,
                'rmsd_lb': rmsd_lb,
                'rmsd_ub': rmsd_ub,
                'cluster_size': len(cluster),
                'pose': best_pose
            })
        
        # Sort by affinity
        return sorted(results, key=lambda r: r['affinity'])


class VinaScoringFunction:
    """
    Physics-based scoring function inspired by Vina
    Can be used alongside or instead of neural network
    """
    
    def __init__(self, protein_coords, atom_types):
        self.protein_coords = protein_coords
        self.atom_types = atom_types
        
        # Atom type parameters (simplified)
        self.radii = {
            'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8,
            'H': 1.2, 'F': 1.47, 'Cl': 1.75, 'Br': 1.85
        }
        
        self.is_hydrophobic = {'C', 'S'}
        self.is_hbond_donor = {'N', 'O'}
        self.is_hbond_acceptor = {'N', 'O'}
    
    def __call__(self, pose: Pose) -> float:
        """
        Calculate total binding energy
        """
        
        score = 0.0
        
        # Van der Waals
        score += self.vdw_energy(pose.coordinates)
        
        # Hydrogen bonds
        score += self.hbond_energy(pose.coordinates)
        
        # Hydrophobic effect
        score += self.hydrophobic_energy(pose.coordinates)
        
        # Torsion penalty
        score += self.torsion_penalty(pose.torsions)
        
        return score
    
    def vdw_energy(self, coords: np.ndarray) -> float:
        """
        Soft van der Waals (Gaussian-based, no singularities)
        """
        
        energy = 0.0
        
        for i, atom_i in enumerate(coords):
            for j, atom_j in enumerate(self.protein_coords):
                if j >= i:
                    break
                    
                r = np.linalg.norm(atom_i - atom_j)
                
                if r < 0.5:
                    r = 0.5  # Prevent singularity
                
                # Gaussian repulsion + attraction
                energy += 0.5 * np.exp(-(r - 4.0)**2 / 2.0)  # Attraction
                energy += 0.5 * np.exp(-(r - 8.0)**2 / 2.0)   # Repulsion
        
        return energy
    
    def hbond_energy(self, coords: np.ndarray) -> float:
        """
        Directional hydrogen bonding
        """
        
        energy = 0.0
        
        for i, atom_i in enumerate(coords):
            for j, atom_j in enumerate(self.protein_coords):
                r = np.linalg.norm(atom_i - atom_j)
                
                # H-bond favorable range
                if 1.9 <= r <= 3.5:
                    # Simplified: just distance-based
                    energy -= 1.5 * np.exp(-(r - 2.0)**2 / 0.25)
        
        return energy
    
    def hydrophobic_energy(self, coords: np.ndarray) -> float:
        """
        Hydrophobic effect (favorable at ~5Å)
        """
        
        energy = 0.0
        
        for i, atom_i in enumerate(coords):
            for j, atom_j in enumerate(self.protein_coords):
                r = np.linalg.norm(atom_i - atom_j)
                
                energy -= 0.3 * np.exp(-(r - 5.0)**2 / 1.0)
        
        return energy
    
    def torsion_penalty(self, torsions: np.ndarray) -> float:
        """
        Entropy cost of fixing rotatable bonds
        """
        
        return 0.6 * len(torsions)


def vina_style_docking(
    initial_pose: Pose,
    scoring_function,
    exhaustiveness: int = 8,
    max_steps: int = 100,
) -> List[Dict]:
    """
    Complete Vina-style docking pipeline
    
    Returns ranked poses with energies
    """
    
    optimizer = VinaStyleOptimizer(scoring_function)
    
    # Run search
    all_poses = optimizer.docking_search(
        initial_pose,
        exhaustiveness=exhaustiveness,
        max_steps=max_steps
    )
    
    if not all_poses:
        return []
    
    # Cluster
    clusters = PoseClustering.cluster_poses(all_poses, rmsd_threshold=2.0)
    
    # Rank
    results = PoseClustering.rank_clusters(clusters)
    
    return results
