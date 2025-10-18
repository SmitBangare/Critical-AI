from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ClinicalConfig:
    hidden_size: int = 768
    dropout: float = 0.1


class ClinicalDataProcessor(nn.Module):
    """
    Processes clinical/respiratory data into embeddings compatible with multi-modal attention.
    """
    def __init__(self, config: ClinicalConfig):
        super().__init__()
        self.config = config
        
        # Respiratory parameter embeddings
        self.respiratory_rate_embedding = nn.Linear(1, config.hidden_size)
        self.oxygen_sat_embedding = nn.Linear(1, config.hidden_size)
        self.blood_pressure_sys_embedding = nn.Linear(1, config.hidden_size)
        self.blood_pressure_dia_embedding = nn.Linear(1, config.hidden_size)
        
        # Ventilator parameter embeddings
        self.tidal_volume_embedding = nn.Linear(1, config.hidden_size)
        self.peep_embedding = nn.Linear(1, config.hidden_size)
        self.peak_pressure_embedding = nn.Linear(1, config.hidden_size)
        
        # Blood gas embeddings
        self.ph_embedding = nn.Linear(1, config.hidden_size)
        self.pao2_embedding = nn.Linear(1, config.hidden_size)
        self.paco2_embedding = nn.Linear(1, config.hidden_size)
        
        # Demographics embeddings
        self.age_embedding = nn.Linear(1, config.hidden_size)
        self.sex_embedding = nn.Linear(1, config.hidden_size)
        
        # Fusion layer to combine all clinical features
        self.fusion_layer = nn.Linear(config.hidden_size * 12, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, clinical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            clinical_data: Dictionary containing clinical parameters
                Expected keys: respiratory_rate, oxygen_saturation, blood_pressure_sys,
                              blood_pressure_dia, tidal_volume, peep, peak_pressure,
                              ph, pao2, paco2, age, sex
        Returns:
            clinical_features: [B, hidden_size] - Processed clinical embeddings
        """
        batch_size = next(iter(clinical_data.values())).shape[0]
        device = next(iter(clinical_data.values())).device
        
        # Initialize embeddings list
        embeddings = []
        
        # Process respiratory parameters
        if 'respiratory_rate' in clinical_data:
            rr_emb = self.respiratory_rate_embedding(clinical_data['respiratory_rate'].unsqueeze(-1))
            embeddings.append(rr_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'oxygen_saturation' in clinical_data:
            o2_emb = self.oxygen_sat_embedding(clinical_data['oxygen_saturation'].unsqueeze(-1))
            embeddings.append(o2_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'blood_pressure_sys' in clinical_data:
            bp_sys_emb = self.blood_pressure_sys_embedding(clinical_data['blood_pressure_sys'].unsqueeze(-1))
            embeddings.append(bp_sys_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'blood_pressure_dia' in clinical_data:
            bp_dia_emb = self.blood_pressure_dia_embedding(clinical_data['blood_pressure_dia'].unsqueeze(-1))
            embeddings.append(bp_dia_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
        
        # Process ventilator parameters
        if 'tidal_volume' in clinical_data:
            tv_emb = self.tidal_volume_embedding(clinical_data['tidal_volume'].unsqueeze(-1))
            embeddings.append(tv_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'peep' in clinical_data:
            peep_emb = self.peep_embedding(clinical_data['peep'].unsqueeze(-1))
            embeddings.append(peep_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'peak_pressure' in clinical_data:
            pp_emb = self.peak_pressure_embedding(clinical_data['peak_pressure'].unsqueeze(-1))
            embeddings.append(pp_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
        
        # Process blood gas parameters
        if 'ph' in clinical_data:
            ph_emb = self.ph_embedding(clinical_data['ph'].unsqueeze(-1))
            embeddings.append(ph_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'pao2' in clinical_data:
            pao2_emb = self.pao2_embedding(clinical_data['pao2'].unsqueeze(-1))
            embeddings.append(pao2_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'paco2' in clinical_data:
            paco2_emb = self.paco2_embedding(clinical_data['paco2'].unsqueeze(-1))
            embeddings.append(paco2_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
        
        # Process demographics
        if 'age' in clinical_data:
            age_emb = self.age_embedding(clinical_data['age'].unsqueeze(-1))
            embeddings.append(age_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
            
        if 'sex' in clinical_data:
            sex_emb = self.sex_embedding(clinical_data['sex'].unsqueeze(-1))
            embeddings.append(sex_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.config.hidden_size, device=device))
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=-1)  # [B, hidden_size * 12]
        
        # Fuse into single clinical representation
        clinical_features = self.fusion_layer(all_embeddings)
        clinical_features = self.dropout(clinical_features)
        
        return clinical_features


def create_sample_clinical_data(batch_size: int = 2, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Create sample clinical data for testing purposes.
    """
    return {
        'respiratory_rate': torch.randn(batch_size, device=device) * 10 + 20,  # 20-30 breaths/min
        'oxygen_saturation': torch.randn(batch_size, device=device) * 5 + 95,   # 95-100%
        'blood_pressure_sys': torch.randn(batch_size, device=device) * 20 + 120,  # 120 mmHg
        'blood_pressure_dia': torch.randn(batch_size, device=device) * 10 + 80,   # 80 mmHg
        'tidal_volume': torch.randn(batch_size, device=device) * 100 + 500,      # 500 ml
        'peep': torch.randn(batch_size, device=device) * 2 + 5,                  # 5 cmH2O
        'peak_pressure': torch.randn(batch_size, device=device) * 5 + 20,         # 20 cmH2O
        'ph': torch.randn(batch_size, device=device) * 0.1 + 7.4,                # 7.4
        'pao2': torch.randn(batch_size, device=device) * 20 + 100,                # 100 mmHg
        'paco2': torch.randn(batch_size, device=device) * 5 + 40,                 # 40 mmHg
        'age': torch.randn(batch_size, device=device) * 20 + 60,                  # 60 years
        'sex': torch.randint(0, 2, (batch_size,), device=device).float(),        # 0/1
    }
