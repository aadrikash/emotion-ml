"""
Decision engine: determines what action and when for each emotional state.
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class DecisionEngine:
    """Rule-based decision system for wellness recommendations."""
    
    # Action mappings based on emotion + intensity + context
    ACTION_MAPPING = {
        'calm': {
            'low': {'high_energy': 'deep_work', 'low_energy': 'light_planning'},
            'high': {'high_energy': 'deep_work', 'low_energy': 'journaling'},
        },
        'anxious': {
            'low': {'high_energy': 'movement', 'low_energy': 'grounding'},
            'high': {'high_energy': 'box_breathing', 'low_energy': 'box_breathing'},
        },
        'content': {
            'low': {'high_energy': 'movement', 'low_energy': 'light_planning'},
            'high': {'high_energy': 'deep_work', 'low_energy': 'journaling'},
        },
        'restless': {
            'low': {'high_energy': 'movement', 'low_energy': 'grounding'},
            'high': {'high_energy': 'movement', 'low_energy': 'yoga'},
        },
        'overwhelmed': {
            'low': {'high_energy': 'pause', 'low_energy': 'rest'},
            'high': {'high_energy': 'box_breathing', 'low_energy': 'rest'},
        },
        'focused': {
            'low': {'high_energy': 'light_planning', 'low_energy': 'journaling'},
            'high': {'high_energy': 'deep_work', 'low_energy': 'deep_work'},
        },
        'neutral': {
            'low': {'high_energy': 'light_planning', 'low_energy': 'rest'},
            'high': {'high_energy': 'movement', 'low_energy': 'grounding'},
        },
        'sad': {
            'low': {'high_energy': 'sound_therapy', 'low_energy': 'rest'},
            'high': {'high_energy': 'yoga', 'low_energy': 'rest'},
        },
        'excited': {
            'low': {'high_energy': 'movement', 'low_energy': 'journaling'},
            'high': {'high_energy': 'deep_work', 'low_energy': 'light_planning'},
        },
        'frustrated': {
            'low': {'high_energy': 'movement', 'low_energy': 'pause'},
            'high': {'high_energy': 'box_breathing', 'low_energy': 'rest'},
        },
        'mixed': {
            'low': {'high_energy': 'pause', 'low_energy': 'grounding'},
            'high': {'high_energy': 'yoga', 'low_energy': 'sound_therapy'},
        }
    }
    
    # Timing logic based on state, intensity, time of day, and stress
    TIMING_LOGIC = {
        'now': [
            # High stress + high intensity + daytime = immediate action
            lambda row: row['stress_level'] >= 4 and row['intensity'] >= 4 and row['time_of_day'] in ['morning', 'afternoon'],
            # Anxious/restless + high intensity = immediate
            lambda row: row['emotional_state'] in ['anxious', 'restless'] and row['intensity'] >= 4,
            # Very low energy = rest now
            lambda row: row['energy_level'] <= 1 and row['emotional_state'] in ['overwhelmed', 'sad'],
        ],
        'within_15_min': [
            # Moderate stress + medium intensity
            lambda row: 2 <= row['stress_level'] <= 3 and 2 <= row['intensity'] <= 3,
            # Restless + evening/night
            lambda row: row['emotional_state'] == 'restless' and row['time_of_day'] in ['evening', 'night'],
            # Low stress + low-medium intensity
            lambda row: row['stress_level'] <= 1 and row['intensity'] <= 3,
        ],
        'later_today': [
            # Calm/neutral + low intensity
            lambda row: row['emotional_state'] in ['calm', 'neutral', 'content'] and row['intensity'] <= 2,
            # Afternoon time + moderate stress
            lambda row: row['time_of_day'] == 'afternoon' and row['stress_level'] <= 2,
        ],
        'tonight': [
            # Evening/night time + moderate states
            lambda row: row['time_of_day'] in ['evening', 'night'] and row['intensity'] <= 3,
            # High energy + evening = plan for night
            lambda row: row['energy_level'] >= 4 and row['time_of_day'] == 'evening',
        ],
        'tomorrow_morning': [
            # Night time + low intensity
            lambda row: row['time_of_day'] == 'night' and row['intensity'] <= 2,
            # Overwhelmed + late evening = defer to morning
            lambda row: row['emotional_state'] == 'overwhelmed' and row['time_of_day'] == 'night',
        ]
    }
    
    def decide_action(self, row: pd.Series) -> str:
        """Determine what action to take."""
        emotional_state = row['emotional_state']
        intensity = row['intensity']
        energy = row['energy_level']
        stress = row['stress_level']
        
        # Get intensity level (low/high based on median)
        intensity_level = 'high' if intensity >= 3 else 'low'
        
        # Get energy level
        energy_level = 'high_energy' if energy >= 3 else 'low_energy'
        
        # Default mapping
        if emotional_state in self.ACTION_MAPPING:
            try:
                action = self.ACTION_MAPPING[emotional_state][intensity_level][energy_level]
            except:
                action = 'pause'  # Safe default
        else:
            action = 'pause'
        
        # Override logic based on stress
        if stress >= 5 and intensity >= 4:
            if energy <= 2:
                action = 'rest'
            else:
                action = 'box_breathing'
        
        return action
    
    def decide_timing(self, row: pd.Series) -> str:
        """Determine when to take action."""
        # Check each timing condition
        for timing in ['now', 'within_15_min', 'later_today', 'tonight', 'tomorrow_morning']:
            conditions = self.TIMING_LOGIC[timing]
            if any(condition(row) for condition in conditions):
                return timing
        
        # Default
        if row['time_of_day'] == 'night':
            return 'tomorrow_morning'
        elif row['time_of_day'] in ['evening', 'night']:
            return 'tonight'
        else:
            return 'later_today'
    
    def generate_message(self, row: pd.Series, action: str, timing: str) -> str:
        """Generate supportive message."""
        emotional_state = row['emotional_state']
        intensity = row['intensity']
        
        # Map emotions to supportive messages
        messages = {
            'anxious': f"You seem a bit anxious. Let's ground you with {action.replace('_', ' ')}.",
            'overwhelmed': f"You're carrying a lot. Let's pause and breathe. Try {action.replace('_', ' ')} soon.",
            'restless': f"Your mind is active. Channel that energy into {action.replace('_', ' ')}.",
            'calm': f"You're in a good place. Use this clarity for {action.replace('_', ' ')}.",
            'sad': f"Let's take care of you. {action.replace('_', ' ').title()} might help.",
            'focused': f"You're locked in. Keep the momentum with {action.replace('_', ' ')}.",
            'neutral': f"A small shift can help. Try {action.replace('_', ' ')}.",
            'content': f"You're in a good space. {action.replace('_', ' ').title()} would complement this.",
            'excited': f"Channel this energy! {action.replace('_', ' ').title()} would be perfect.",
            'frustrated': f"Take a break. {action.replace('_', ' ').title()} will help you reset.",
            'mixed': f"You're in two places. Let's find balance with {action.replace('_', ' ')}.",
        }
        
        base_msg = messages.get(emotional_state, f"Try {action.replace('_', ' ')}.")
        
        # Add timing context
        timing_map = {
            'now': 'Right now is the best time.',
            'within_15_min': 'Do it in the next 15 minutes.',
            'later_today': 'Try it later today.',
            'tonight': "You can do this tonight.",
            'tomorrow_morning': "Let's start fresh tomorrow morning."
        }
        
        return base_msg + " " + timing_map.get(timing, "")
    
    def decide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make decisions for all rows."""
        decisions = []
        
        for idx, row in df.iterrows():
            action = self.decide_action(row)
            timing = self.decide_timing(row)
            message = self.generate_message(row, action, timing)
            
            decisions.append({
                'what_to_do': action,
                'when_to_do': timing,
                'supportive_message': message
            })
        
        return pd.DataFrame(decisions)