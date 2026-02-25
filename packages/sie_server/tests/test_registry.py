from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.deps import DepConflict, DependencyConflictError
from sie_server.core.registry import ModelRegistry


def _make_config(
    name: str = "test",
    hf_id: str | None = "org/test",
    dense_dim: int = 768,
    max_sequence_length: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=dense_dim))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
        max_sequence_length=max_sequence_length,
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    """Patch ensure_model_cached to avoid actual HF downloads in tests."""
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


class TestModelRegistry:
    """Tests for ModelRegistry."""

    @pytest.fixture(autouse=True)
    def patch_check_deps(self) -> MagicMock:
        """Patch check_model_dependencies for all tests."""
        with patch("sie_server.core.registry.check_model_dependencies", return_value=[]) as mock:
            yield mock

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.capabilities.outputs = ["dense"]
        return mock

    def test_empty_registry(self) -> None:
        """Can create empty registry."""
        registry = ModelRegistry()

        assert registry.model_names == []
        assert registry.loaded_model_names == []

    def test_add_config(self) -> None:
        """Can add config programmatically."""
        registry = ModelRegistry()

        config = _make_config(name="test-model", hf_id="org/test")

        registry.add_config(config)

        assert registry.has_model("test-model")
        assert registry.get_config("test-model") == config

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Can load configs from directory."""
        # Create flat YAML config file
        (tmp_path / "my-model.yaml").write_text("""
sie_id: my-model
hf_id: org/my-model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: "sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter"
    max_batch_tokens: 8192
""")

        registry = ModelRegistry(models_dir=tmp_path)

        assert registry.has_model("my-model")
        assert registry.get_config("my-model").tasks.encode.dense.dim == 384

    def test_cloud_models_dir_maps_cached_configs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cloud models_dir maps each model to the cache directory (flat YAML structure)."""
        cache_root = tmp_path / "cache"
        monkeypatch.setenv("SIE_LOCAL_CACHE", str(cache_root))
        cache_dir = cache_root / "sie_configs"
        cache_dir.mkdir(parents=True)

        def write_cached_config(model_name: str, dense_dim: int) -> Path:
            # Flat YAML structure: configs are directly in cache_dir
            yaml_filename = model_name.lower().replace("/", "-") + ".yaml"
            config_path = cache_dir / yaml_filename
            config_path.write_text(
                "\n".join(
                    [
                        f"sie_id: {model_name}",
                        f"hf_id: {model_name}",
                        "tasks:",
                        "  encode:",
                        "    dense:",
                        f"      dim: {dense_dim}",
                        "profiles:",
                        "  default:",
                        "    adapter_path: sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                        "    max_batch_tokens: 8192",
                        "",
                    ]
                )
            )
            return config_path

        model_a = "org/model-a"
        model_b = "org/model-b"
        write_cached_config(model_a, 384)
        write_cached_config(model_b, 768)

        configs = {
            model_a: _make_config(name=model_a, hf_id=model_a, dense_dim=384),
            model_b: _make_config(name=model_b, hf_id=model_b),
        }

        monkeypatch.setattr("sie_server.core.registry.load_model_configs", lambda _: configs)
        monkeypatch.setattr("sie_sdk.storage.is_cloud_path", lambda _: True)

        registry = ModelRegistry(models_dir="s3://bucket/models")

        # With flat YAML structure, all models share the same cache dir as their model_dir
        assert registry._model_dirs[model_a] == cache_dir
        assert registry._model_dirs[model_b] == cache_dir

    def test_has_model(self) -> None:
        """has_model returns correct values."""
        registry = ModelRegistry()

        config = _make_config(name="exists")
        registry.add_config(config)

        assert registry.has_model("exists") is True
        assert registry.has_model("nonexistent") is False

    def test_is_loaded(self) -> None:
        """is_loaded returns correct values."""
        registry = ModelRegistry()

        config = _make_config(name="test")
        registry.add_config(config)

        assert registry.is_loaded("test") is False

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can load a model."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        adapter = registry.load("test", device="cpu")

        assert adapter is mock_adapter
        mock_adapter.load.assert_called_once_with("cpu")
        assert registry.is_loaded("test")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_loaded_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can get a loaded model's adapter."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        adapter = registry.get("test")

        assert adapter is mock_adapter

    def test_get_unloaded_model_raises(self) -> None:
        """Get raises for unloaded model."""
        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        with pytest.raises(KeyError, match="is not loaded"):
            registry.get("test")

    def test_get_unknown_model_raises(self) -> None:
        """Get raises for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_already_loaded_raises(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Load raises if model already loaded."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        with pytest.raises(ValueError, match="already loaded"):
            registry.load("test", device="cpu")

    def test_load_unknown_model_raises(self) -> None:
        """Load raises for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.load("nonexistent", device="cpu")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_model(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can unload a model."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        registry.unload("test")

        mock_adapter.unload.assert_called_once()
        assert not registry.is_loaded("test")

    def test_unload_not_loaded_raises(self) -> None:
        """Unload raises if model not loaded."""
        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        with pytest.raises(KeyError, match="is not loaded"):
            registry.unload("test")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_all(self, mock_load_adapter: MagicMock) -> None:
        """Can unload all models."""
        mock_adapter_a = MagicMock()
        mock_adapter_b = MagicMock()
        mock_load_adapter.side_effect = [mock_adapter_a, mock_adapter_b]

        registry = ModelRegistry()

        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        assert len(registry.loaded_model_names) == 2

        registry.unload_all()

        assert len(registry.loaded_model_names) == 0
        mock_adapter_a.unload.assert_called_once()
        mock_adapter_b.unload.assert_called_once()

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_model_info(self, mock_load_adapter: MagicMock, mock_adapter: MagicMock) -> None:
        """Can get model info."""
        mock_load_adapter.return_value = mock_adapter

        registry = ModelRegistry()
        config = _make_config(name="test", max_sequence_length=512)
        registry.add_config(config)

        # Before loading
        info = registry.get_model_info("test")
        assert info["name"] == "test"
        assert info["loaded"] is False
        assert info["device"] is None
        assert info["dims"]["dense"] == 768

        # After loading
        registry.load("test", device="cuda:0")
        info = registry.get_model_info("test")
        assert info["loaded"] is True
        assert info["device"] == "cuda:0"


class TestRegistryMemoryManagerIntegration:
    """Tests for ModelRegistry + MemoryManager integration (LRU eviction)."""

    @pytest.fixture(autouse=True)
    def patch_check_deps(self) -> MagicMock:
        """Patch check_model_dependencies for all tests."""
        with patch("sie_server.core.registry.check_model_dependencies", return_value=[]) as mock:
            yield mock

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters."""

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            return mock

        return make_mock

    @patch("sie_server.core.model_loader.load_adapter")
    def test_load_registers_with_memory_manager(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Loading a model registers it with the memory manager."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")

        # Model should be registered in memory manager
        assert registry.memory_manager.loaded_model_count == 1
        assert "test" in registry.memory_manager.loaded_models

    @patch("sie_server.core.model_loader.load_adapter")
    def test_unload_unregisters_from_memory_manager(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Unloading a model unregisters it from the memory manager."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)
        registry.load("test", device="cpu")
        registry.unload("test")

        # Model should be unregistered from memory manager
        assert registry.memory_manager.loaded_model_count == 0
        assert "test" not in registry.memory_manager.loaded_models

    @patch("sie_server.core.model_loader.load_adapter")
    def test_get_touches_model_for_lru(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Getting a model's adapter updates LRU tracking."""
        mock_load_adapter.side_effect = [mock_adapter_factory(), mock_adapter_factory()]

        registry = ModelRegistry()

        # Add and load two models
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Initially model-a is LRU (loaded first)
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access model-a, now model-b should be LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"

        # Access model-b, now model-a should be LRU again
        registry.get("model-b")
        assert registry.memory_manager.get_lru_model() == "model-a"

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_triggers_lru_eviction_and_retry(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """OOM during load triggers LRU eviction and retry."""
        # First two loads succeed
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()

        # Third adapter fails with OOM on first try
        adapter_c_fail = mock_adapter_factory()
        oom_error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        adapter_c_fail.load.side_effect = oom_error

        # Fourth adapter (retry) succeeds
        adapter_c_success = mock_adapter_factory()

        # Side effect: a, b, c_fail, c_success (retry creates new adapter)
        mock_load_adapter.side_effect = [adapter_a, adapter_b, adapter_c_fail, adapter_c_success]

        registry = ModelRegistry()

        # Add three model configs
        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        # Load first two models
        registry.load("model-a", device="cuda:0")
        registry.load("model-b", device="cuda:0")

        # model-a is LRU
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Load third model - should trigger OOM, evict model-a, then succeed on retry
        registry.load("model-c", device="cuda:0")

        # model-a should be evicted
        assert not registry.is_loaded("model-a")
        adapter_a.unload.assert_called_once()

        # model-c should be loaded (via retry adapter)
        assert registry.is_loaded("model-c")
        # First adapter failed, retry adapter succeeded
        adapter_c_fail.load.assert_called_once()
        adapter_c_success.load.assert_called_once()

        # Now only model-b and model-c are loaded
        assert len(registry.loaded_model_names) == 2
        assert set(registry.loaded_model_names) == {"model-b", "model-c"}

    @patch("sie_server.core.model_loader.load_adapter")
    def test_oom_with_no_models_to_evict_raises(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """OOM with no models to evict raises the original error."""
        adapter = mock_adapter_factory()
        adapter.load.side_effect = RuntimeError("CUDA out of memory")

        mock_load_adapter.return_value = adapter

        registry = ModelRegistry()
        config = _make_config(name="test")
        registry.add_config(config)

        # No models loaded, so no LRU to evict - should raise
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            registry.load("test", device="cuda:0")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_non_oom_error_propagates(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Non-OOM RuntimeError propagates without eviction attempt."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        adapter_b.load.side_effect = RuntimeError("Some other error")

        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        registry.load("model-a", device="cpu")

        # Non-OOM error should propagate without evicting model-a
        with pytest.raises(RuntimeError, match="Some other error"):
            registry.load("model-b", device="cpu")

        # model-a should still be loaded (no eviction attempted)
        assert registry.is_loaded("model-a")
        adapter_a.unload.assert_not_called()


class TestRegistryOOMDetection:
    """Tests for _is_oom_error detection."""

    def test_cuda_oom_detected(self) -> None:
        """CUDA OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert registry._is_oom_error(error) is True

    def test_mps_oom_detected(self) -> None:
        """MPS OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("MPS backend out of memory")
        assert registry._is_oom_error(error) is True

    def test_generic_oom_detected(self) -> None:
        """Generic OOM error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("Cannot allocate memory for tensor")
        assert registry._is_oom_error(error) is True

    def test_allocation_failed_detected(self) -> None:
        """Allocation failed error is detected."""
        registry = ModelRegistry()

        error = RuntimeError("Failed to allocate 8GB")
        assert registry._is_oom_error(error) is True

    def test_non_oom_not_detected(self) -> None:
        """Non-OOM error is not detected as OOM."""
        registry = ModelRegistry()

        error = RuntimeError("Some other error")
        assert registry._is_oom_error(error) is False

    def test_case_insensitive_detection(self) -> None:
        """OOM detection is case insensitive."""
        registry = ModelRegistry()

        error = RuntimeError("OUT OF MEMORY - CUDA")
        assert registry._is_oom_error(error) is True


class TestMultiModelRouting:
    """Tests for multi-model routing (Project 4.3).

    Verifies:
    - Each loaded model has its own ModelWorker
    - Requests can alternate between multiple loaded models
    - Correct adapter is returned for each model
    """

    @pytest.fixture(autouse=True)
    def patch_check_deps(self) -> MagicMock:
        """Patch check_model_dependencies for all tests."""
        with patch("sie_server.core.registry.check_model_dependencies", return_value=[]) as mock:
            yield mock

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters with unique IDs."""
        counter = [0]

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            counter[0] += 1
            mock._test_id = counter[0]  # Unique ID for verification
            return mock

        return make_mock

    @patch("sie_server.core.model_loader.load_adapter")
    def test_each_model_has_own_worker(self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock) -> None:
        """Each loaded model has its own ModelWorker instance."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        # Add two model configs
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Get workers for both models
        worker_a = registry.get_worker("model-a")
        worker_b = registry.get_worker("model-b")

        # Each model has its own worker
        assert worker_a is not None
        assert worker_b is not None
        assert worker_a is not worker_b

        # Workers reference correct adapters
        assert worker_a.adapter is adapter_a
        assert worker_b.adapter is adapter_b

    @patch("sie_server.core.model_loader.load_adapter")
    def test_alternating_requests_to_different_models(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Can alternate requests between multiple loaded models."""
        adapter_a = mock_adapter_factory()
        adapter_b = mock_adapter_factory()
        mock_load_adapter.side_effect = [adapter_a, adapter_b]

        registry = ModelRegistry()

        # Add and load two models
        for name in ["model-a", "model-b"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Alternate requests between models multiple times
        for _ in range(3):
            # Request to model-a
            got_adapter_a = registry.get("model-a")
            assert got_adapter_a is adapter_a
            assert got_adapter_a._test_id == 1

            # Request to model-b
            got_adapter_b = registry.get("model-b")
            assert got_adapter_b is adapter_b
            assert got_adapter_b._test_id == 2

        # Both models still loaded after alternating requests
        assert registry.is_loaded("model-a")
        assert registry.is_loaded("model-b")

    @patch("sie_server.core.model_loader.load_adapter")
    def test_correct_adapter_returned_by_model_name(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Requests are routed to the correct model adapter by name."""
        # Create 3 models with distinct adapters
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        registry = ModelRegistry()

        model_names = ["model-alpha", "model-beta", "model-gamma"]
        for name in model_names:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Request models in random order and verify correct routing
        assert registry.get("model-gamma")._test_id == 3
        assert registry.get("model-alpha")._test_id == 1
        assert registry.get("model-beta")._test_id == 2
        assert registry.get("model-alpha")._test_id == 1  # Same result on repeat
        assert registry.get("model-gamma")._test_id == 3

    @patch("sie_server.core.model_loader.load_adapter")
    def test_worker_not_available_for_unloaded_model(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Worker returns None for models that aren't loaded."""
        mock_load_adapter.return_value = mock_adapter_factory()

        registry = ModelRegistry()

        config = _make_config(name="test-model")
        registry.add_config(config)

        # Worker should be None before loading
        assert registry.get_worker("test-model") is None

        # Load the model
        registry.load("test-model", device="cpu")

        # Worker should be available after loading
        assert registry.get_worker("test-model") is not None

        # Unload
        registry.unload("test-model")

        # Worker should be None again
        assert registry.get_worker("test-model") is None

    @patch("sie_server.core.model_loader.load_adapter")
    def test_lru_updates_on_alternating_access(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """LRU tracking updates correctly when alternating between models."""
        mock_load_adapter.side_effect = [mock_adapter_factory() for _ in range(3)]

        registry = ModelRegistry()

        # Load 3 models in order: A, B, C
        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)
            registry.load(name, device="cpu")

        # Initially A is LRU (loaded first)
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access A -> now B is LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"

        # Access B -> now C is LRU
        registry.get("model-b")
        assert registry.memory_manager.get_lru_model() == "model-c"

        # Access C -> now A is LRU (full rotation)
        registry.get("model-c")
        assert registry.memory_manager.get_lru_model() == "model-a"

        # Access A again -> now B is LRU
        registry.get("model-a")
        assert registry.memory_manager.get_lru_model() == "model-b"


class TestAsyncLoading:
    """Tests for async model loading (DESIGN.md Section 5.4)."""

    @pytest.fixture(autouse=True)
    def patch_check_deps(self) -> MagicMock:
        """Patch check_model_dependencies for all tests."""
        with patch("sie_server.core.registry.check_model_dependencies", return_value=[]) as mock:
            yield mock

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.capabilities.outputs = ["dense"]
        mock.memory_footprint.return_value = 1_000_000
        return mock

    @pytest.fixture
    def registry_with_model(self, mock_adapter: MagicMock) -> ModelRegistry:
        """Create registry with a model config ready to load."""
        registry = ModelRegistry()
        config = _make_config(name="test-model", hf_id="org/test")
        registry.add_config(config)
        return registry

    async def test_load_async_basic(self, registry_with_model: ModelRegistry) -> None:
        """Test basic async loading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            adapter = await registry_with_model.load_async("test-model", "cpu")

            assert adapter is mock_adapter
            assert registry_with_model.is_loaded("test-model")
            mock_adapter.load.assert_called_once_with("cpu")

    async def test_load_async_returns_existing_if_loaded(self, registry_with_model: ModelRegistry) -> None:
        """Second call to load_async returns existing model without reloading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            # First load
            adapter1 = await registry_with_model.load_async("test-model", "cpu")
            # Second load should return same adapter without reloading
            adapter2 = await registry_with_model.load_async("test-model", "cpu")

            assert adapter1 is adapter2
            # load_adapter should only be called once
            mock_load.assert_called_once()

    async def test_load_async_concurrent_same_model(self, registry_with_model: ModelRegistry) -> None:
        """Two concurrent loads for same model only load once."""
        import asyncio

        load_count = 0
        load_event = asyncio.Event()

        def slow_load(device: str) -> None:
            nonlocal load_count
            load_count += 1
            # Signal that load started
            load_event.set()

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_adapter.load = slow_load
            mock_load.return_value = mock_adapter

            # Start two concurrent loads
            task1 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))
            task2 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))

            adapter1, adapter2 = await asyncio.gather(task1, task2)

            # Both should return the same adapter
            assert adapter1 is adapter2
            # Load should only happen once
            assert load_count == 1

    async def test_is_unloading_flag(self, registry_with_model: ModelRegistry) -> None:
        """is_unloading returns correct state."""
        assert not registry_with_model.is_unloading("test-model")

    async def test_is_loading_flag_initial_state(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False before load starts."""
        # Model is configured but not loading yet
        assert not registry_with_model.is_loading("test-model")

    async def test_is_loading_flag_cleared_after_load(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False after load completes."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # After load completes, is_loading should be False
            assert not registry_with_model.is_loading("test-model")
            assert registry_with_model.is_loaded("test-model")

    async def test_is_loading_flag_cleared_on_failure(self, registry_with_model: ModelRegistry) -> None:
        """is_loading is cleared even when load fails."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_load.side_effect = RuntimeError("Load failed")

            with pytest.raises(RuntimeError, match="Load failed"):
                await registry_with_model.load_async("test-model", "cpu")

            # After failure, is_loading should still be False
            assert not registry_with_model.is_loading("test-model")
            assert not registry_with_model.is_loaded("test-model")

    async def test_unload_async_drains_worker(self, registry_with_model: ModelRegistry) -> None:
        """unload_async stops worker before unloading adapter."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # Get the worker
            worker = registry_with_model.get_worker("test-model")
            assert worker is not None

            # Start the worker
            await registry_with_model.start_worker("test-model")

            # Now unload
            await registry_with_model.unload_async("test-model")

            assert not registry_with_model.is_loaded("test-model")
            mock_adapter.unload.assert_called_once()

    async def test_unload_all_async(self, registry_with_model: ModelRegistry) -> None:
        """unload_all_async unloads all models."""
        # Add another model
        config2 = _make_config(name="model-2", hf_id="org/test2")
        registry_with_model.add_config(config2)

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")
            await registry_with_model.load_async("model-2", "cpu")

            assert len(registry_with_model.loaded_model_names) == 2

            await registry_with_model.unload_all_async()

            assert len(registry_with_model.loaded_model_names) == 0

    async def test_load_async_model_not_found_raises(self) -> None:
        """load_async raises KeyError for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            await registry.load_async("unknown-model", "cpu")

    async def test_load_async_while_unloading_raises(self, registry_with_model: ModelRegistry) -> None:
        """load_async raises RuntimeError if model is being unloaded."""
        # Manually set unloading flag
        registry_with_model._unloading.add("test-model")

        with pytest.raises(RuntimeError, match="currently being unloaded"):
            await registry_with_model.load_async("test-model", "cpu")


class TestProactiveEviction:
    """Tests for proactive memory eviction (pre-load and background monitor)."""

    @pytest.fixture(autouse=True)
    def patch_check_deps(self) -> MagicMock:
        """Patch check_model_dependencies for all tests."""
        with patch("sie_server.core.registry.check_model_dependencies", return_value=[]) as mock:
            yield mock

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters."""

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            mock.memory_footprint.return_value = 1000
            return mock

        return make_mock

    @pytest.fixture
    def registry_with_models(self, mock_adapter_factory: MagicMock) -> ModelRegistry:
        """Create registry with 3 model configs."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(pressure_threshold=0.85),
        )

        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        return registry

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_triggers_when_above_threshold(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction evicts LRU when memory is above threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        assert registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")

        # Mock memory pressure (90% usage, threshold is 85%)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, False]):
            # Load third model - should trigger eviction of model-a (LRU)
            await registry_with_models.load_async("model-c", "cpu")

        # model-a should be evicted, model-b and model-c loaded
        assert not registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_loop_evicts_multiple(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction can evict multiple models until below threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        # Mock memory pressure that requires evicting both models
        # check_pressure: True, True, False (evict a, evict b, then ok)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, True, False]):
            await registry_with_models.load_async("model-c", "cpu")

        # Both model-a and model-b should be evicted
        assert not registry_with_models.is_loaded("model-a")
        assert not registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    async def test_background_monitor_loop_runs(self) -> None:
        """Background monitor loop runs and checks pressure periodically."""
        import asyncio

        from sie_server.core.memory import MemoryConfig

        # Create registry with short check interval for testing
        registry = ModelRegistry(
            memory_config=MemoryConfig(
                pressure_threshold=0.85,
                memory_check_interval_s=0.005,  # 5ms for fast testing
            ),
        )

        # Track how many times check_pressure is called
        check_count = 0

        def counting_check() -> bool:
            nonlocal check_count
            check_count += 1
            return False  # No pressure, so no eviction needed

        registry._memory_manager.check_pressure = counting_check

        await registry.start_memory_monitor()
        try:
            # Wait for a few check cycles
            await asyncio.sleep(0.02)
            # Should have been called multiple times
            assert check_count >= 2
        finally:
            await registry.stop_memory_monitor()

    async def test_memory_monitor_starts_and_stops(self) -> None:
        """Memory monitor can be started and stopped cleanly."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(memory_check_interval_s=0.1),
        )

        assert registry._monitor_task is None
        assert not registry._monitor_running

        await registry.start_memory_monitor()

        assert registry._monitor_task is not None
        assert registry._monitor_running

        await registry.stop_memory_monitor()

        assert registry._monitor_task is None
        assert not registry._monitor_running

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_adapter_unload_called_on_unload(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Adapter.unload() is called when model is unloaded.

        Memory cleanup (gc.collect + empty_cache) is the adapter's responsibility.
        See the memory management contract in ModelAdapter docstring (base.py).
        """
        mock_adapter = mock_adapter_factory()
        mock_load_adapter.return_value = mock_adapter

        await registry_with_models.load_async("model-a", "cuda:0")
        await registry_with_models.unload_async("model-a")

        mock_adapter.unload.assert_called_once()


class TestDependencyConflictError:
    """Tests for dependency checking during model loading.

    Verifies that DependencyConflictError is raised synchronously
    (before background loading starts) when model dependencies conflict.
    """

    @pytest.fixture
    def registry_with_model(self) -> ModelRegistry:
        """Create registry with a model config."""
        registry = ModelRegistry()
        config = _make_config(name="test-model", hf_id="org/test")
        registry.add_config(config)
        return registry

    @pytest.fixture
    def conflict(self) -> DepConflict:
        """Create a sample dependency conflict."""
        return DepConflict(
            package="transformers",
            required=">=4.45,<4.54",
            installed="4.57.3",
            reason="version mismatch",
        )

    def test_check_model_loadable_raises_on_conflict(
        self, registry_with_model: ModelRegistry, conflict: DepConflict
    ) -> None:
        """_check_model_loadable raises DependencyConflictError when deps conflict."""
        with patch(
            "sie_server.core.registry.check_model_dependencies",
            return_value=[conflict],
        ):
            with pytest.raises(DependencyConflictError) as exc_info:
                registry_with_model._check_model_loadable("test-model")

            assert exc_info.value.model_name == "test-model"
            assert len(exc_info.value.conflicts) == 1
            assert exc_info.value.conflicts[0].package == "transformers"

    def test_load_raises_on_conflict(self, registry_with_model: ModelRegistry, conflict: DepConflict) -> None:
        """Sync load() raises DependencyConflictError before loading."""
        with patch(
            "sie_server.core.registry.check_model_dependencies",
            return_value=[conflict],
        ):
            with pytest.raises(DependencyConflictError) as exc_info:
                registry_with_model.load("test-model", device="cpu")

            assert "transformers" in str(exc_info.value)

    async def test_load_async_raises_on_conflict(
        self, registry_with_model: ModelRegistry, conflict: DepConflict
    ) -> None:
        """Async load_async() raises DependencyConflictError before loading."""
        with patch(
            "sie_server.core.registry.check_model_dependencies",
            return_value=[conflict],
        ):
            with pytest.raises(DependencyConflictError) as exc_info:
                await registry_with_model.load_async("test-model", device="cpu")

            assert exc_info.value.model_name == "test-model"

    async def test_start_load_async_raises_on_conflict(
        self, registry_with_model: ModelRegistry, conflict: DepConflict
    ) -> None:
        """start_load_async() raises DependencyConflictError synchronously.

        This is the key test: errors must surface before background task is created,
        so the API can return 409 instead of 503.
        """
        with patch(
            "sie_server.core.registry.check_model_dependencies",
            return_value=[conflict],
        ):
            with pytest.raises(DependencyConflictError) as exc_info:
                await registry_with_model.start_load_async("test-model", device="cpu")

            # Error raised synchronously before background task
            assert exc_info.value.model_name == "test-model"
            # Model should NOT be in loading state (error raised before marking)
            assert not registry_with_model.is_loading("test-model")

    async def test_start_load_async_success_without_conflicts(self, registry_with_model: ModelRegistry) -> None:
        """start_load_async() succeeds when no dependency conflicts."""
        import asyncio

        with patch(
            "sie_server.core.registry.check_model_dependencies",
            return_value=[],  # No conflicts
        ):
            with patch("sie_server.core.model_loader.load_adapter") as mock_load:
                mock_adapter = MagicMock()
                mock_adapter.memory_footprint.return_value = 1000
                mock_load.return_value = mock_adapter

                # Should return True (load started)
                result = await registry_with_model.start_load_async("test-model", device="cpu")

                assert result is True
                # Model should be in loading state
                assert registry_with_model.is_loading("test-model")

                # Wait for background task to complete to avoid "Task was destroyed" warning
                while registry_with_model.is_loading("test-model"):  # noqa: ASYNC110
                    await asyncio.sleep(0.01)
