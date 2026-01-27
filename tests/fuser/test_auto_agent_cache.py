# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from Fuser.auto_agent import _validate_cfg_models


class TestValidateCfgModels:
    def test_validate_cfg_models_removes_unavailable(self):
        """Test that unavailable models are removed from config."""
        cfg = {
            "ka_model": "gpt-5",
            "llm_models": {
                "extract": "gpt-5",
                "dispatch": "o4-mini",
            },
        }

        with patch("Fuser.auto_agent.is_model_available", return_value=False):
            _validate_cfg_models(cfg)

        assert "ka_model" not in cfg
        assert "llm_models" not in cfg

    def test_validate_cfg_models_keeps_available(self):
        """Test that available models are kept in config."""
        cfg = {
            "ka_model": "gpt-5",
            "llm_models": {
                "extract": "gpt-5",
                "dispatch": "o4-mini",
            },
        }

        with patch("Fuser.auto_agent.is_model_available", return_value=True):
            _validate_cfg_models(cfg)

        assert cfg["ka_model"] == "gpt-5"
        assert cfg["llm_models"]["extract"] == "gpt-5"
        assert cfg["llm_models"]["dispatch"] == "o4-mini"
