"""
Fetch model weights from HF
(requires prior access)
Sean Browning
"""

import os
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download

DEFAULT_CA_BUNDLE = Path("/etc/ssl/cert.pem")
CHECKPOINT_DIR = Path("checkpoints") / "CONCH"
REVISION = "f9ca9f877171a28ade80228fb195ac5d79003357"


def configure_ca_bundle(ca_path: Path = DEFAULT_CA_BUNDLE):
    """
    Helper to handle CA Cert bundle on WSL2
    """
    # httpx/certifi may not pick up WSL's system trust store by default.
    for env_var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(env_var, str(ca_path))


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, default=CHECKPOINT_DIR, help="Path to folder where model weights will be stored")
    parser.add_argument("--revision", type=str, default=REVISION, help="(optional) Sha1 of the revision to pull")
    parser.add_argument("--ca_bundle_path", type=Path, default=DEFAULT_CA_BUNDLE, help="(optional) path to CA Cert bundle")
    args = parser.parse_args()

    load_dotenv()

    if args.ca_bundle_path:
        configure_ca_bundle(args.ca_bundle_path)


    # Ensure our directory exists before we run things
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    model_bin_path = hf_hub_download(
        repo_id="MahmoodLab/CONCH",
        filename="pytorch_model.bin",
        revision=args.revision,
        local_dir=CHECKPOINT_DIR,
        token=os.getenv("HF_API_KEY"),
    )

    print(f"Model bin downloaded to: {model_bin_path}")


if __name__ == "__main__":
    main()
