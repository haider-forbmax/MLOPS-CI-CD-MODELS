#!/usr/bin/env python3
"""
Standalone Triton client utility to inspect model config metadata and extract classes.

Usage:
  python triton_classes_client.py --url 192.168.18.30:3087 --model yolo5-ticker-flasher
"""

import argparse
import ast
import json
from typing import Any, Dict

import tritonclient.http as httpclient


def _parameter_string_value(parameters: Any, key_name: str) -> str:
    """Read parameter value for key from dict-style or list-style Triton responses."""
    if isinstance(parameters, dict):
        param = parameters.get(key_name)
        if isinstance(param, dict):
            return param.get("string_value") or param.get("stringValue", "")
        if isinstance(param, str):
            return param
        return ""

    if isinstance(parameters, list):
        for item in parameters:
            if not isinstance(item, dict):
                continue
            if item.get("key") != key_name:
                continue
            value = item.get("value", {})
            if isinstance(value, dict):
                return value.get("string_value") or value.get("stringValue", "")
            if isinstance(value, str):
                return value
        return ""

    return ""


def _parse_metadata_blob(blob: str) -> Dict[str, Any]:
    """Parse metadata blob that may be Python-literal or strict JSON."""
    if not blob:
        return {}

    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(blob)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    return {}


def _extract_classes_from_response(resp: Dict[str, Any]) -> Dict[str, str]:
    """Extract classes from Triton get_model_config/get_model_metadata response."""
    if not isinstance(resp, dict):
        return {}

    payload = resp.get("config", resp)
    params = payload.get("parameters", {})
    blob = _parameter_string_value(params, "metadata")
    parsed = _parse_metadata_blob(blob)
    names = parsed.get("names", {})

    if not isinstance(names, dict):
        return {}
    return {str(k): str(v) for k, v in names.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Triton model metadata/classes for ticker-flasher model.")
    parser.add_argument("--url", default="192.168.18.30:3087", help="Triton HTTP URL in host:port format")
    parser.add_argument("--model", default="yolo5-ticker-flasher", help="Model name")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout seconds")
    args = parser.parse_args()

    client = httpclient.InferenceServerClient(url=args.url, connection_timeout=args.timeout)

    print(f"Triton URL: {args.url}")
    print(f"Model: {args.model}")

    try:
        is_live = client.is_server_live()
        is_ready = client.is_server_ready()
        model_ready = client.is_model_ready(args.model)
        print(f"Live={is_live} Ready={is_ready} ModelReady={model_ready}")
    except Exception as exc:
        print(f"Health check failed: {exc}")
        return 1

    classes: Dict[str, str] = {}

    try:
        cfg = client.get_model_config(model_name=args.model)
        cfg_classes = _extract_classes_from_response(cfg)
        print(f"Config response type: {type(cfg).__name__}")
        if cfg_classes:
            print(f"Classes from model config: {len(cfg_classes)}")
            classes = cfg_classes
        else:
            print("Classes not found in model config parameters.metadata")
    except Exception as exc:
        print(f"get_model_config failed: {exc}")

    try:
        meta = client.get_model_metadata(model_name=args.model)
        meta_classes = _extract_classes_from_response(meta)
        print(f"Metadata response type: {type(meta).__name__}")
        if meta_classes:
            print(f"Classes from model metadata: {len(meta_classes)}")
            if not classes:
                classes = meta_classes
        else:
            print("Classes not found in model metadata parameters.metadata")
    except Exception as exc:
        print(f"get_model_metadata failed: {exc}")

    if not classes:
        print("No classes extracted from Triton responses.")
        return 2

    print("\nExtracted classes JSON:")
    print(json.dumps(classes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
