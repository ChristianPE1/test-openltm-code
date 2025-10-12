# 🌊 ENSO Phase Analysis Strategy for Timer-XL Thesis

## 📊 Current Results Summary

### Model Performance (5 Years ERA5: 2020-2024)

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | VRAM |
|-------|----------|-----------|--------|----------|---------------|------|
| **Small Model** (4 layers) | **76.93%** | **0.87** | **0.70** | **0.78** | 15 min | 1.5 GB |
| Big Model (scratch) | 57.98% | - | - | - | 40 min (incomplete) | ~6 GB |
| **Transfer Learning** | **76.23%** (epoch 4) | - | - | - | 40+ min (running) | ~8 GB |

**Key Insight**: 
- Small Model provides excellent baseline (F1=0.78) with minimal resources
- Transfer Learning shows promise (improving trend: 64% → 73% → 76%)
- High precision (0.87) means low false alarm rate for rain prediction

---

## 🎯 Thesis Contribution Strategy

Based on advisor guidance: **"Reproducible Timer-XL pipeline for ENSO-influenced rainfall prediction"**

### Decision Tree:

```
Transfer Learning F1-Score?
├─ F1 > 0.80 (Good)
│  └─ Focus: "Optimal Context Length Analysis for ENSO Prediction"
│     ├─ Contribution: First systematic study of context requirements
│     ├─ Experiments: 5 context lengths × 3 ENSO phases
│     └─ Deliverable: Context length guidelines + reproducible pipeline
│
└─ F1 < 0.75 (Poor)
   └─ Focus: "Timer-XL Architecture Improvements for Climate Prediction"
      ├─ Contribution: ENSO-aware attention mechanism
      ├─ Experiments: Modified architecture + ablation studies
      └─ Deliverable: Enhanced Timer-XL + reproducible pipeline
```

---

## 🌍 ENSO Phase Analysis Methodology

### Step 1: Define ENSO Phases (2020-2024)

Use **Oceanic Niño Index (ONI)** to classify months:
- **El Niño**: ONI ≥ +0.5°C for 3+ consecutive months
- **La Niña**: ONI ≤ -0.5°C for 3+ consecutive months
- **Neutral**: -0.5°C < ONI < +0.5°C

**Your 5-Year Data Coverage:**
```
2020: Jan-Aug (Neutral), Sep-Dec (La Niña)
2021: Jan-Apr (La Niña), May-Dec (Neutral)
2022: Jan-May (Neutral), Jun-Dec (La Niña)
2023: Jan-May (La Niña), Jun-Dec (Neutral → El Niño)
2024: Jan-May (El Niño), Jun-Dec (Neutral)
```

**Approximate Distribution:**
- El Niño: ~8 months (17%)
- La Niña: ~20 months (42%)
- Neutral: ~20 months (41%)

### Step 2: Data Splitting by ENSO Phase

**Option A: Temporal Validation (Recommended for Climate)**
```python
# Maintain temporal order (climate patterns)
Train: 2020-2022 (3 years)
Val:   2023 (1 year)
Test:  2024 (1 year)

# Then evaluate metrics per ENSO phase
F1_ElNino = evaluate(test_samples[phase == 'El Niño'])
F1_LaNina = evaluate(test_samples[phase == 'La Niña'])
F1_Neutral = evaluate(test_samples[phase == 'Neutral'])
```

**Option B: Phase-Stratified Split (More balanced)**
```python
# Split each phase independently (70/15/15)
Train: 70% of El Niño + 70% La Niña + 70% Neutral
Val:   15% of El Niño + 15% La Niña + 15% Neutral
Test:  15% of El Niño + 15% La Niña + 15% Neutral
```

**Recommendation**: Use **Option A** for thesis (more realistic for climate prediction)

### Step 3: Performance Metrics per ENSO Phase

For each model and context length:

```python
metrics = {
    'El Niño': {
        'Accuracy': ...,
        'Precision': ...,
        'Recall': ...,
        'F1-Score': ...,
        'Confusion Matrix': ...
    },
    'La Niña': {...},
    'Neutral': {...}
}
```

**Success Criteria:**
1. **Consistency**: `|F1_ElNiño - F1_LaNiña| < 0.15` (no phase bias)
2. **Minimum Performance**: `F1 > 0.70` for all phases
3. **Recall for El Niño**: `Recall > 0.75` (critical for flood prediction)

---

## 🔬 Context Length Experiments

### Hypothesis
*"Longer context windows capture ENSO teleconnections (90-120 day cycles), improving prediction accuracy during extreme phases"*

### Experimental Design

**Context Lengths to Test:**
1. **90 days** (3 months) - Minimum ENSO definition period
2. **180 days** (6 months) - One ENSO season
3. **365 days** (1 year) - Full seasonal cycle
4. **730 days** (2 years) - Multi-year ENSO transitions
5. **1095 days** (3 years) - Long-term climate memory

**For Each Context Length:**
```python
for seq_len in [90*24, 180*24, 365*24, 730*24, 1095*24]:  # Convert days to hours
    model = train_model(seq_len=seq_len)
    
    for phase in ['El Niño', 'La Niña', 'Neutral']:
        metrics = evaluate_phase(model, phase)
        results[seq_len][phase] = metrics
```

**Analysis:**
1. **Saturation Point**: Where improvement < 2% with longer context
2. **Phase-Specific Optima**: Does El Niño need more context than Neutral?
3. **Cost-Benefit**: Training time vs accuracy gain

**Expected Results:**
```
Context Length | F1-Score | Training Time | Saturation?
---------------|----------|---------------|------------
90 days        | 0.68     | 10 min        | No
180 days       | 0.74     | 15 min        | No
365 days       | 0.78     | 20 min        | Maybe
730 days       | 0.79     | 35 min        | Yes (< 2% gain)
1095 days      | 0.79     | 60 min        | Yes (no gain)
```

**Thesis Contribution:**
> *"We identify 365-730 days as the optimal context length for ENSO-influenced rainfall prediction, capturing full seasonal cycles while avoiding diminishing returns beyond 2-year windows."*

---

## 📈 Visualization Plan for Thesis

### Figure 1: Model Comparison
```python
# Bar chart: Accuracy, Precision, Recall, F1 for 3 models
models = ['Small', 'Transfer Learning', 'Big (scratch)']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
```

### Figure 2: ENSO Phase Performance
```python
# Grouped bar chart: F1-Score per phase for each model
phases = ['El Niño', 'La Niña', 'Neutral']
models = ['Small', 'Transfer Learning']
```

### Figure 3: Context Length Analysis
```python
# Line plot: F1-Score vs Context Length (by ENSO phase)
x = [90, 180, 365, 730, 1095]  # days
y_elnino = [...]
y_lanina = [...]
y_neutral = [...]
```

### Figure 4: Confusion Matrices
```python
# 3x3 grid: Confusion matrices for best model across 3 ENSO phases
```

### Figure 5: Training Efficiency
```python
# Scatter plot: Training Time vs F1-Score (context length as color)
```

---

## 🛠️ Implementation Roadmap

### Phase 1: Checkpoint Evaluation (Today)
- [ ] Run `test_checkpoint_standalone.py --find_latest`
- [ ] Get Transfer Learning F1-Score
- [ ] Compare Small vs Transfer Learning
- [ ] **Decision**: Context length experiments vs Architecture improvements

### Phase 2: ENSO Data Preparation (1 day)
- [ ] Download ONI index (2020-2024)
- [ ] Create `enso_phase_split.py` to label each sample
- [ ] Validate phase distribution matches historical records
- [ ] Add ENSO phase column to `peru_rainfall_cleaned.csv`

### Phase 3: Context Length Experiments (3-5 days)
**If Transfer Learning F1 > 0.75:**
- [ ] Create `context_length_experiments.py`
- [ ] Train 5 models with different seq_len
- [ ] Evaluate each on 3 ENSO phases
- [ ] Plot saturation curves
- [ ] Write methodology section

### Phase 4: Architecture Improvements (Optional, 5-7 days)
**If Transfer Learning F1 < 0.75:**
- [ ] Design ENSO-aware attention mechanism
- [ ] Implement multi-scale temporal modeling
- [ ] Run ablation studies
- [ ] Compare with baseline Timer-XL

### Phase 5: Thesis Writing (Ongoing)
- [ ] Methodology: Reproducible pipeline documentation
- [ ] Results: Tables + figures for each experiment
- [ ] Discussion: Optimal context length recommendations
- [ ] Conclusion: Contributions to Timer-XL for climate prediction

---

## 📝 Scripts Needed

### 1. `enso_phase_split.py`
Download ONI index and label each sample with ENSO phase

### 2. `context_length_experiments.py`
Automated training for 5 context lengths with phase-specific evaluation

### 3. `enso_metrics_analysis.py`
Calculate and compare metrics across ENSO phases

### 4. `plot_thesis_results.py`
Generate all 5 figures for thesis

### 5. `reproducible_pipeline.py`
End-to-end script: data cleaning → training → ENSO evaluation

---

## 🎓 Thesis Structure Outline

### Chapter 1: Introduction
- Motivation: ENSO impact on Peru rainfall
- Problem: Existing models don't leverage long context
- Contribution: Optimal context length for ENSO prediction

### Chapter 2: Literature Review
- ENSO teleconnections and rainfall prediction
- Transformer models for time series (Timer-XL, Timer, etc.)
- Context length in climate forecasting

### Chapter 3: Methodology
- **3.1 Data**: ERA5 2020-2024, ENSO phase labeling
- **3.2 Model**: Timer-XL architecture + transfer learning
- **3.3 Experiments**: 5 context lengths × 3 ENSO phases
- **3.4 Evaluation**: Phase-specific metrics, saturation analysis

### Chapter 4: Results
- **4.1 Model Comparison**: Small vs Big vs Transfer Learning
- **4.2 ENSO Performance**: F1-Score per phase
- **4.3 Context Length**: Saturation point identification
- **4.4 Case Studies**: El Niño 2023-2024 predictions

### Chapter 5: Discussion
- **5.1 Optimal Context**: 365-730 days for ENSO prediction
- **5.2 Phase Differences**: Why El Niño needs more context
- **5.3 Practical Guidelines**: When to use which context length

### Chapter 6: Conclusions
- Reproducible Timer-XL pipeline established
- Context length recommendations for ENSO-affected regions
- Future work: Architecture improvements, multi-region validation

---

## 🚀 Next Steps (Immediate)

1. **Test Transfer Learning Checkpoint** (10 minutes)
   ```bash
   python test_checkpoint_standalone.py --find_latest
   ```

2. **Decide Thesis Path** (Based on F1-Score)
   - F1 > 0.80: Context length experiments (safer, reproducible)
   - F1 < 0.75: Architecture improvements (riskier, more novel)

3. **Create ENSO Phase Dataset** (2 hours)
   - Download ONI index from NOAA
   - Label each sample with phase
   - Validate distribution

4. **Run First Context Length Experiment** (3 hours)
   - Test seq_len = 365 days (baseline)
   - Evaluate on 3 ENSO phases
   - Validate methodology works

5. **Write Methodology Chapter** (Ongoing)
   - Document reproducible pipeline
   - Explain ENSO phase evaluation
   - Describe context length experiments

---

## 📧 Questions for Advisor (Next Meeting)

1. **Thesis Scope**: Should I focus on context length OR architecture improvements?
2. **ENSO Phases**: Use temporal validation (Option A) or stratified split (Option B)?
3. **Context Lengths**: Are 5 experiments enough? Or test more granular (120, 240, 480 days)?
4. **Evaluation**: Is F1 > 0.70 for all phases a reasonable success criterion?
5. **Contribution Claim**: "First systematic context length study for ENSO prediction with Timer-XL"?

---

## 🎯 Success Criteria (Thesis Acceptance)

1. ✅ **Reproducible Pipeline**: Anyone can replicate results with provided code
2. ✅ **Clear Contribution**: Optimal context length identified (e.g., 365-730 days)
3. ✅ **Validation**: F1 > 0.70 on all 3 ENSO phases
4. ✅ **Consistency**: No phase bias (|F1_diff| < 0.15)
5. ✅ **Documentation**: Complete methodology + code repository

---

## 📚 References to Include

1. **ENSO**: Trenberth (1997) - ENSO definition
2. **Timer-XL**: Original Timer-XL paper (2024)
3. **ERA5**: Hersbach et al. (2020) - ERA5 dataset
4. **Context Length**: Dosovitskiy et al. (2021) - Vision Transformers (context scaling)
5. **Climate Prediction**: Reichstein et al. (2019) - Deep learning for Earth system

---

**Created**: 2025-01-11  
**Last Updated**: 2025-01-11  
**Status**: Strategy finalized, awaiting checkpoint test results
