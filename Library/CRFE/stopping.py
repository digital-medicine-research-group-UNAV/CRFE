import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from functools import lru_cache


@dataclass
class ParamParada:
    """Parámetros para el criterio de parada del CRFE."""
    alpha: float = 0.05        # cuantil para robustez
    eps: float = 0.02          # mejora relativa mínima (2%) para "mejora material"
    eta: float = 0.10          # retroceso absoluto que dispara parada inmediata
    paciencia: int = 580         # iteraciones sin mejora material antes de chequear dominancia
    usar_dominancia: bool = False  # si usar dominancia de Pareto en parada
    tau: float = 0.05          # umbral para dominancia


@lru_cache(maxsize=128)
def _cached_eps_calculation(median_val: float, eps_factor: float = 1e-9) -> float:
    """Cache para el cálculo de epsilon."""
    return max(median_val * eps_factor, 1e-12)



def _normalizar_por_radii(D: np.ndarray, eps_factor: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normaliza D por filas usando r_i = max(D_ii, eps). Devuelve D_tilde = R^{-1} D.
    """
    D = np.asarray(D, dtype=np.float64)
    
    # Extraer diagonal
    diag = np.diag(D)
    
    # Calcular eps de forma eficiente
    mask_pos = np.isfinite(diag) & (diag > 0)
    if np.any(mask_pos):
        base = float(np.median(diag[mask_pos]))
    else:
        pos_vals = D[np.isfinite(D) & (D > 0)]
        base = float(np.median(pos_vals)) if pos_vals.size > 0 else 1.0
    
    eps = _cached_eps_calculation(base, eps_factor)
    
    # Clamp radios usando operaciones vectorizadas
    r = np.where(mask_pos, diag, eps)
    
    # Normalización vectorizada
    D_tilde = D / r[:, np.newaxis]
    
    # Fijar diagonal
    np.fill_diagonal(D_tilde, 1.0)
    
    return D_tilde, r, eps


@dataclass
class Margenes:
    rho_min: float           # min off-diagonal de D_tilde
    rho_q: float             # cuantil bajo (robusto)
    rho_star: float          # min(rho_min, rho_q)
    gamma_min: float         # rho_min - 1
    D_tilde: np.ndarray      # matriz normalizada usada


class StoppingCriteria:
    
    def __init__(self, params: ParamParada = None):
        self.params = params if params is not None else ParamParada()
        self.n_features = None
        self.reset()

    def reset(self):
        """Reinicia el estado interno."""
        self.mejor_val = -np.inf
        self.mejor_idx = -1
        self.mejor_D_tilde = None
        self.rachas_sin_mejora = 0
        self.t = -1
        self.historial = []

    @staticmethod
    def compute_proximity_matrix( X, y):
        """
        Computes proximity matrix (n_clases x n_clases) where:
        - [i,j] = distancia mínima del centroide de clase i a observaciones de clase j (i!=j)
        - [i,i] = distancia máxima del centroide de clase i a observaciones de clase i
        """
        

        if hasattr(X, 'values'):  # pandas DataFrame
            X = X.values
        if hasattr(y, 'values'):  # pandas Series
            y = y.values

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        unique_classes, y_indices = np.unique(y, return_inverse=True)
        n_clases = len(unique_classes)
        
        if n_clases < 2:
            return np.array([[1.0]])

        # Calcular centroides vectorizadamente
        centroides = np.array([X[y == cls].mean(axis=0) for cls in unique_classes])
        
        # Matriz resultado
        D = np.zeros((n_clases, n_clases))

        # Para cada centroide de referencia
        for cls_idx in range(n_clases):
            pos_centroide = centroides[cls_idx]
            
            # Distancias de todas las observaciones a este centroide
            distancias_todas = np.linalg.norm(X - pos_centroide, axis=1)
            
            # Para cada clase objetivo
            for target_cls_idx in range(n_clases):
                mask_clase = (y_indices == target_cls_idx)
                if not np.any(mask_clase):
                    D[cls_idx, target_cls_idx] = 0.0
                    continue
                    
                dists_clase = distancias_todas[mask_clase]
                
                if target_cls_idx == cls_idx:
                    # Misma clase: 5% mas lejano (dispersión intra-clase)
                    D[cls_idx, target_cls_idx] = np.quantile(dists_clase, 0.95) 
                else:
                    # Clase diferente: 5% mas cercano (separación inter-clase)
                    D[cls_idx, target_cls_idx] = np.quantile(dists_clase, 0.05)

        return D

    def matrix_evaluation(self, D: np.ndarray, alpha: float = None):
        if alpha is None:
            alpha = self.params.alpha
            
        D_tilde, r, eps = _normalizar_por_radii(D)

        # We get off-diagonal elements
        mask_off = ~np.eye(D_tilde.shape[0], dtype=bool)
        off_vals = D_tilde[mask_off] # extract and plain them
        off_finite = off_vals[np.isfinite(off_vals)] # check no Nans or inf

        if off_finite.size == 0:
            rho_min = rho_q = -np.inf
        else:
            rho_min = float(np.min(off_finite))
            rho_q = float(np.quantile(off_finite, max(0.0, min(1.0, alpha))))

        rho_star = min(rho_min, rho_q)
        gamma_min = rho_min - 1.0
        
        return Margenes(
            rho_min=rho_min, rho_q=rho_q, rho_star=rho_star,
            gamma_min=gamma_min, D_tilde=D_tilde
        )
    
    def _auto_eta_from_history(self, W: int = 20, alpha: float = None,
                               eps_eta: float = 1e-3) -> float:
        
        
        if alpha is None:
            alpha = self.params.alpha

        # Si no hay historial suficiente -> no activar retroceso
        if len(self.historial) < 5:
            return float('inf')

        # Usar historial anterior a la iteración actual
        vals = np.array([h['rho_star'] for h in self.historial[:-1]], dtype=float)
        vals = vals[-W:]  # ventana deslizante

        if vals.size == 0 or not np.isfinite(vals).any():
            return float('inf')

        # Drawdowns respecto al máximo acumulado
        running_max = np.maximum.accumulate(vals)
        drawdowns = np.clip(running_max - vals, 0.0, np.inf)
        fin = drawdowns[np.isfinite(drawdowns)]
        if fin.size == 0:
            return float('inf')

        mad = np.median(np.abs(fin - np.median(fin))) * 1.4826  # escala robusta
        qhi = np.quantile(fin, 1.0 - alpha)

        eta = max(float(qhi), 3.0 * float(mad), eps_eta)
        return eta
        
    def update(self, X, y):
        """
        Actualiza el criterio de parada con nuevos datos.
        
        Returns:
            tuple: (debe_parar, mejor_idx, motivo)
        """
        self.t += 1
        D = self.compute_proximity_matrix(X, y)
        m = self.matrix_evaluation(D)

        #print(D)
        
        
        print("rho_star:", m.rho_star)
        # Save step
        self.historial.append({
            'iter': self.t,
            'rho_star': m.rho_star,
            'rho_min': m.rho_min
        })

        if self.t == 0:
            self.mejor_val = m.rho_star
            self.mejor_idx = 0
            self.mejor_D_tilde = m.D_tilde.copy()
            return False, self.mejor_idx, None

        # Umbralización precomputada
        umbral_mejora = self.mejor_val * (1.0 + 0.005) 

        eta = 0.12 #= self._auto_eta_from_history(W=5, alpha=self.params.alpha, eps_eta=1e-3)
        umbral_retroceso = self.mejor_val - eta
        print("\neta: ", eta)

        print("umbral mejora:", umbral_mejora)
        print("umbral retroceso:", umbral_retroceso)

        if m.rho_star >= umbral_mejora:  # Mejora  detectada
            
            print(f"Mejora detectada en iteración {self.t}: {m.rho_star} (anterior: {self.mejor_val}) \n")
            
            self.mejor_val = m.rho_star
            self.mejor_idx = self.t
            self.mejor_D_tilde = m.D_tilde.copy()
            #self.rachas_sin_mejora = 0
            return False, self.mejor_idx, "mejora_detectada"
        
        if m.rho_star < umbral_retroceso:
                # Retroceso significativo: parar inmediatamente
                print(f"Stopping at iteration {self.t} due to retrocess")
                return True, self.mejor_idx, "retroceso_significativo"
    
        

        # Sin mejora material
        #self.rachas_sin_mejora += 1

        #if self.rachas_sin_mejora >= self.params.paciencia:
            # Se agotó la paciencia
            #if self.params.usar_dominancia:
                # Verificar dominancia de Pareto
            #    if not self._domina(m.D_tilde, self.mejor_D_tilde, self.params.tau):
            #        return True, self.mejor_idx, "sin_mejora_y_no_domina"
            #    else:
                    # Hay dominancia, continuar
            #        return False, self.mejor_idx, "dominancia_detectada"
            #else:
            # Sin dominancia, parar por paciencia
        #    return True, self.mejor_idx, "paciencia_agotada"
                
        # Continuar
        return False, self.mejor_idx, "continuar"

    #def _domina(self, D1: np.ndarray, D2: np.ndarray, tau: float) -> bool:
    #    """
    #    Verifica si D1 domina a D2 en el sentido de Pareto con tolerancia tau.
    #    """
        # Extraer elementos off-diagonal
    #    mask_off = ~np.eye(D1.shape[0], dtype=bool)
    #    off1 = D1[mask_off]
    #    off2 = D2[mask_off]
        
        # Filtrar valores finitos
    #    mask_finite = np.isfinite(off1) & np.isfinite(off2)
    #    if not np.any(mask_finite):
    #        return False
            
    #    off1_clean = off1[mask_finite]
    #    off2_clean = off2[mask_finite]
        
        # D1 domina a D2 si: todos los elementos >= (con tolerancia) Y al menos uno es estrictamente mayor
    #    diff = off1_clean - off2_clean
    #    todos_ge = np.all(diff >= -tau)
    #    alguno_mayor = np.any(diff > tau)
        
    #    return todos_ge and alguno_mayor
