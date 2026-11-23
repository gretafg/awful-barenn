import subprocess
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ShellSyntax(unittest.TestCase):
    def test_all_shell_and_slurm_files_parse(self):
        for source in sorted((ROOT/"scripts").iterdir()):
            if source.suffix in (".sh",".slurm"):
                with self.subTest(source=source.name):
                    subprocess.run(["bash","-n",str(source)],check=True,capture_output=True)

if __name__=="__main__":
    unittest.main()
