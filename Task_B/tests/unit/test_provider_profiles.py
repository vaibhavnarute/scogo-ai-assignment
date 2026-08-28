from harness.providers.nvidia import DEFAULT_NVIDIA_MODEL, NVIDIA_API_KEY_ENV, NVIDIA_BASE_URL


def test_nvidia_defaults_are_explicit():
    assert DEFAULT_NVIDIA_MODEL == "openai/gpt-oss-20b"
    assert NVIDIA_API_KEY_ENV == "NVIDIA_API_KEY"
    assert NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"