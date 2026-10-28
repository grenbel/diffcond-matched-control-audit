# ODGNet files for the UpTrans experiments

The cross-backbone experiments with UpTrans (Section 5.5 of the paper) use two files from the ODGNet repository (Cai et al., *Orthogonal Dictionary Guided Shape Completion Network for Point Cloud*, AAAI 2024): `models/pointnet.py` and `models/UpTrans.py`. The ODGNet repository does not state a licence, so this repository does not include either file. It provides the commit we used, the SHA-256 of both files, and our five-line change to `models/UpTrans.py` as a patch.

`models/__init__.py` imports both files, so they are needed before any experiment in this repository is run, the AdaPoinTr ones included.

## Automatic

From the repository root:

```bash
python third_party/fetch_odgnet.py
```

The script downloads both files from ODGNet commit `032bca98f89cc3ea4733b331ff91468df1bea0f9`, checks their SHA-256, applies `odgnet_uptrans.patch`, checks the result against the files used in our runs and writes them to `models/`. With `--check-only` it runs the checks without writing anything; with `--from-dir DIR` it uses copies you have already downloaded to `DIR`.

## Manual

1. Download `models/pointnet.py` and `models/UpTrans.py` from https://github.com/corecai163/ODGNet at commit `032bca98f89cc3ea4733b331ff91468df1bea0f9` (for example https://raw.githubusercontent.com/corecai163/ODGNet/032bca98f89cc3ea4733b331ff91468df1bea0f9/models/UpTrans.py), keep their LF line endings, and put them in `models/`.
2. From the repository root, run `git -c core.autocrlf=false apply --unidiff-zero third_party/odgnet_uptrans.patch` or `patch -p1 < third_party/odgnet_uptrans.patch`. The patch carries no context lines, so it quotes only the five ODGNet lines it replaces; `git apply` needs `--unidiff-zero` for such a patch, and `core.autocrlf=false` stops it from writing Windows line endings.
3. Compare the SHA-256 of both files with the right-hand column below.

| File | ODGNet commit `032bca9` (SHA-256, LF) | As used in our runs (SHA-256) |
|---|---|---|
| `models/pointnet.py` | `26f4925535e46702145af8d4e06e30384e189770e78c50aeeb995a03a78e5ab7` | the same file, unchanged |
| `models/UpTrans.py` | `a72033030560556f1dfef8b1d0a1b686d8fa4367326f61f909d52e5560ce2e34` | `0e8baead8cf8c8b12750ebbc236c36847278d2edd3a89c8d2cb49db2e6927a17` |

The right-hand column equals the hashes recorded for these two files in `data/results/gap_a_provenance/meta/code_attestation.json`.

## What the patch changes

Five lines of `models/UpTrans.py`: the import of `models.pointnet` becomes a relative import that aliases `Transformer` as `UpTransTransformer`; three lines use that name (one of them is commented out); one shape comment is corrected.

## Credit

ODGNet belongs to its authors. If you use these files, please cite:

```bibtex
@inproceedings{odgnet,
  title={Orthogonal Dictionary Guided Shape Completion Network for Point Cloud},
  author={Cai, Pingping and Scott, Deja and Li, Xiaoguang and Wang, Song},
  booktitle={AAAI},
  volume={38},
  pages={864--872},
  year={2024},
  doi={10.1609/aaai.v38i2.27845}
}
```
