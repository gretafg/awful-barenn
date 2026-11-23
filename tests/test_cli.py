import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from compare import compare

class Acceptance(unittest.TestCase):
    def test_complete_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for backend in ("legacy","seq","omp"):
                output = Path(tmp)/(backend+".json")
                subprocess.run([str(ROOT/"build/barenn_cpu"),"--backend",backend,"--threads","4",
                                "--output",str(output)],cwd=ROOT,check=True,capture_output=True)
                results[backend] = json.loads(output.read_text())
            self.assertEqual(results["legacy"]["correct"],1717)
            for name in ("seq","omp"):
                report = compare(results["legacy"],results[name],atol=0,rtol=0)
                self.assertEqual(report["scores_max_abs"],0)
    def test_invalid_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/"bad.csv"
            bad.write_text("0,1,2,3,invalid\n")
            for args in (["--epochs","0"],["--backend","unknown"],
                         ["--train",str(bad)],
                         ["--train","optdigits.tra","--test","optdigits.tra"]):
                process = subprocess.run([str(ROOT/"build/barenn_cpu"),*args],cwd=ROOT,capture_output=True)
                self.assertNotEqual(process.returncode,0)
    def test_comparator_rejects_nan_and_truncation(self):
        base = dict(epochs=1,train_count=1,test_count=1,epoch_losses=[0.1],
                    scores=[0.2],predictions=[0],confusion=[1])
        for changes in (dict(scores=[float("nan")]),dict(scores=[]),dict(predictions=[1])):
            with self.assertRaises(ValueError):
                compare(base,{**base,**changes})

if __name__ == "__main__":
    unittest.main()
