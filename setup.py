"""
Environment setup and verification script.

Checks:
    - Python version
    - Required packages
    - Dataset availability
    - Checkpoint availability
    - GPU/CUDA status

Usage:
    python setup.py
"""
import sys
import os
import importlib
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def check_python():
    """Check Python version."""
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        print("    ⚠️  Python 3.9+ recommended")
        return False
    print("    ✅ OK")
    return True


def check_packages():
    """Check required packages."""
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'yaml': 'PyYAML',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
    }
    optional = {
        'streamlit': 'Streamlit (demo app)',
        'plotly': 'Plotly (demo charts)',
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {name}: {version} ✅")
        except ImportError:
            print(f"  {name}: NOT INSTALLED ❌")
            all_ok = False

    print("\n  Optional:")
    for pkg, name in optional.items():
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {name}: {version} ✅")
        except ImportError:
            print(f"  {name}: NOT INSTALLED (install with pip)")

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.version.cuda}")
            print(f"  GPU:  {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
            print("    ✅ GPU available")
        else:
            print("  CUDA: Not available (CPU mode)")
            print("    ℹ️  Training requires GPU — use Colab/Kaggle")
        return True
    except Exception as e:
        print(f"  CUDA check failed: {e}")
        return True


def check_datasets():
    """Check dataset availability."""
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset')
    datasets = {
        'ETT-small': ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv'],
        'traffic': ['traffic.csv'],
        'weather': ['weather.csv'],
        'electricity': ['electricity.csv'],
        'solar': ['solar_AL.txt'],
        'PEMS': ['PEMS03.npz', 'PEMS04.npz', 'PEMS07.npz', 'PEMS08.npz'],
        'm4': ['M4-info.csv'],
    }

    all_ok = True
    for ds_name, files in datasets.items():
        ds_path = os.path.join(dataset_dir, ds_name)
        if os.path.isdir(ds_path):
            found = sum(1 for f in files if os.path.exists(os.path.join(ds_path, f)))
            total = len(files)
            status = "✅" if found == total else f"⚠️ ({found}/{total})"
            print(f"  {ds_name}: {status}")
        else:
            print(f"  {ds_name}: NOT FOUND ❌")
            all_ok = False

    return all_ok


def check_checkpoints():
    """Check checkpoint availability."""
    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        print("  No checkpoints directory")
        return False

    checkpoint_count = 0
    datasets_covered = set()

    for entry in os.listdir(ckpt_dir):
        full_path = os.path.join(ckpt_dir, entry)
        if os.path.isdir(full_path):
            pth = os.path.join(full_path, 'checkpoint.pth')
            if os.path.exists(pth):
                checkpoint_count += 1
                # Extract dataset name
                for ds in ['Traffic', 'Weather', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                           'Electricity', 'solar', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'm4']:
                    if ds in entry:
                        datasets_covered.add(ds)
                        break
        elif entry.endswith('.pth'):
            checkpoint_count += 1

    print(f"  Total checkpoints: {checkpoint_count}")
    print(f"  Datasets covered: {', '.join(sorted(datasets_covered))}")
    print(f"    {'✅' if checkpoint_count > 0 else '❌'}")
    return checkpoint_count > 0


def check_baseline_model():
    """Check that baseline model can be imported."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from baseline.models.TimeMixer import Model
        print("  Baseline TimeMixer: ✅ importable")
        return True
    except Exception as e:
        print(f"  Baseline TimeMixer: ❌ {e}")
        return False


def check_extension_model():
    """Check that extension model can be imported."""
    try:
        sys.path.insert(0, PROJECT_ROOT)
        from extension.models.AdaptiveTimeMixer import AdaptiveTimeMixerModel
        print("  Adaptive TimeMixer: ✅ importable")
        return True
    except Exception as e:
        print(f"  Adaptive TimeMixer: ❌ {e}")
        return False


def main():
    print("=" * 60)
    print("  TimeMixer Project — Environment Verification")
    print("=" * 60)

    print("\n📦 Python")
    check_python()

    print("\n📦 Packages")
    check_packages()

    print("\n🖥️ CUDA/GPU")
    check_cuda()

    print("\n📂 Datasets")
    check_datasets()

    print("\n💾 Checkpoints")
    check_checkpoints()

    print("\n🧠 Models")
    check_baseline_model()
    check_extension_model()

    print("\n" + "=" * 60)
    print("  Setup verification complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
