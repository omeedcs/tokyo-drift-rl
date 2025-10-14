# 🚀 START HERE - Quick Navigation Guide

**Welcome to the Autonomous Vehicle Drifting repository!**

This document will get you oriented in **under 2 minutes**.

---

## 🎯 Pick Your Goal

### 1️⃣ "I want to SEE something cool"
```bash
./start_web_ui.sh
```
Opens interactive website at http://localhost:3001 with:
- Live simulation streaming
- Beautiful math equations
- Model comparison (SAC vs IKD)

**Time:** 30 seconds

---

### 2️⃣ "I want to UNDERSTAND the project"
Read these in order:
1. **[Main README](README.md)** - 3-minute overview ⏱️
2. **[PROJECT MAP](PROJECT_MAP.md)** - Visual diagrams 📊
3. **[COMPLETE SUMMARY](COMPLETE_PROJECT_SUMMARY.md)** - Deep dive 📚

**Time:** 10 minutes total

---

### 3️⃣ "I want to REPRODUCE the research"
```bash
source venv/bin/activate
python compare_all_methods.py --trials 20
```

**Time:** 5 minutes (after setup)

**Full guide:** See [Research Section](README.md#-research-experiments)

---

### 4️⃣ "I want to USE the gym environment"
```bash
cd drift_gym
pip install -e .
python scenarios/scenario_generator.py  # Test it works
```

**Time:** 2 minutes

**Full guide:** See [drift_gym/README.md](drift_gym/README.md)

---

### 5️⃣ "I want to TRAIN my own model"

**Train SAC:**
```bash
source venv/bin/activate
python train_sac_simple.py --scenario loose --num_steps 50000
```

**Train IKD:**
```bash
source venv/bin/activate
python collect_ikd_data_corrected.py --episodes 300
python train_ikd_simple.py --data data/ikd_corrected_large.npz
```

**Time:** 8 minutes (SAC), 3 minutes (IKD)

---

## 📁 Project Structure (Simple View)

```
autonomous-vehicle-drifting/
│
├── 🌐 WEBSITE            → ./start_web_ui.sh
│   └── web-ui/
│
├── 🧪 RESEARCH           → python train_sac_simple.py
│   ├── src/
│   ├── trained_models/
│   └── dc_saves/
│
├── 🎮 GYM ENVIRONMENT    → cd drift_gym && pip install -e .
│   └── drift_gym/
│
└── 📚 DOCUMENTATION      → You are here!
    ├── README.md         ← Start here
    ├── PROJECT_MAP.md    ← Visual guide
    └── START_HERE.md     ← You are here
```

---

## 🤔 Common Questions

**Q: Do I need to train models to see the website?**  
A: No! Pre-trained models are included.

**Q: Can I use just the gym environment?**  
A: Yes! `cd drift_gym && pip install -e .`

**Q: How do I activate the virtual environment?**  
A: `source venv/bin/activate`

**Q: Where are all the files?**  
A: See [PROJECT_MAP.md](PROJECT_MAP.md) for complete visual guide.

**Q: What's the difference between the 3 components?**  
A: 
- **Research** = Original experiments (train/test SAC & IKD)
- **Website** = Interactive demo (present your work)  
- **Gym** = Reusable environment (for future research)

---

## 📖 Documentation Hierarchy

```
Level 1: Quick Start
└── START_HERE.md (You are here!) ← 2 min read

Level 2: Overview
├── README.md ← 5 min read
└── PROJECT_MAP.md ← Visual diagrams

Level 3: Component Guides
├── Research: README.md sections
├── Website: WEB_UI_GUIDE.md
└── Gym: drift_gym/README.md

Level 4: Deep Dive
├── COMPLETE_PROJECT_SUMMARY.md
├── DRIFT_GYM_IMPROVEMENTS.md
└── comparison_results/RESULTS.md
```

**Strategy:** Start at Level 1, go deeper as needed!

---

## ⚡ Quick Commands

```bash
# See the demo
./start_web_ui.sh

# Activate environment
source venv/bin/activate

# Train SAC
python train_sac_simple.py

# Compare methods
python compare_all_methods.py --trials 20

# Test gym scenarios
cd drift_gym && python scenarios/scenario_generator.py

# View config
cat drift_gym/config/default_config.yaml
```

---

## 🎯 Next Steps

**After reading this:**

1. ✅ Run `./start_web_ui.sh` to see the website (most impressive!)
2. ✅ Read the [main README](README.md) for full overview
3. ✅ Check [PROJECT_MAP](PROJECT_MAP.md) for visual understanding
4. ✅ Pick your use case and follow that guide

---

## 🆘 Need Help?

- **Visual guide:** [PROJECT_MAP.md](PROJECT_MAP.md)
- **FAQ:** [README.md FAQ section](README.md#-frequently-asked-questions)
- **Issues:** [Open a GitHub issue](https://github.com/omeedcs/autonomous-vehicle-drifting/issues)

---

**You're all set! Pick a goal above and dive in! 🏎️💨**
