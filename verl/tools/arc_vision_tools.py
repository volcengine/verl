# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Arc Vision RL tools for UI element detection enhancement."""

import asyncio
import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ZoomTool(BaseTool):
    """Zoom tool for enhancing UI element visibility.
    
    This tool allows the model to zoom into specific regions of the UI
    when confidence is low for element detection.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_data = {}
        self.default_zoom_factor = config.get("default_zoom_factor", 2.0)
        self.max_zoom_factor = config.get("max_zoom_factor", 4.0)

    async def create(self, instance_id: Optional[str] = None, image: Optional[Image.Image] = None, 
                    confidence: float = 0.5, **kwargs) -> Tuple[str, None]:
        """Initialize tool instance with base image and confidence."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_data[instance_id] = {
            "original_image": image,
            "confidence_before": confidence,
            "zoomed": False,
            "zoom_count": 0,
            "cumulative_confidence_gain": 0.0
        }
        return instance_id, None

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Apply zoom to specified region."""
        if instance_id not in self._instance_data:
            return "Error: Invalid instance ID", -0.1, {}
        
        data = self._instance_data[instance_id]
        
        # Parse parameters
        region = parameters.get("region", [0.25, 0.25, 0.75, 0.75])  # Default to center
        zoom_factor = min(parameters.get("zoom_factor", self.default_zoom_factor), self.max_zoom_factor)
        
        # Validate region bounds
        region = [max(0, min(1, coord)) for coord in region]
        x1, y1, x2, y2 = region
        
        # Apply zoom (simplified - in production this would crop and enhance)
        data["zoomed"] = True
        data["zoom_count"] += 1
        data["last_zoom_region"] = region
        data["last_zoom_factor"] = zoom_factor
        
        # Simulate confidence gain based on zoom factor and region size
        region_size = (x2 - x1) * (y2 - y1)
        base_confidence_gain = 0.2 * zoom_factor * (1 - region_size)  # Smaller regions = higher gain
        
        # Diminishing returns for multiple zooms
        confidence_gain = base_confidence_gain * (0.8 ** (data["zoom_count"] - 1))
        data["cumulative_confidence_gain"] += confidence_gain
        
        # Calculate tool reward
        if confidence_gain > 0.1:
            tool_reward = confidence_gain
        elif data["zoom_count"] > 2:
            tool_reward = -0.2  # Penalty for excessive zooming
        else:
            tool_reward = -0.05  # Small penalty for ineffective zoom
        
        response = f"Zoomed into region {region} with factor {zoom_factor:.1f}. Confidence gain: {confidence_gain:.2f}"
        
        return response, tool_reward, {
            "confidence_gain": confidence_gain,
            "zoom_count": data["zoom_count"],
            "region_size": region_size
        }

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate cumulative reward for tool usage."""
        if instance_id not in self._instance_data:
            return 0.0
        
        data = self._instance_data[instance_id]
        return data["cumulative_confidence_gain"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up instance data."""
        if instance_id in self._instance_data:
            del self._instance_data[instance_id]


class WaitTool(BaseTool):
    """Wait tool for handling loading states and animations.
    
    This tool allows the model to wait for UI elements to stabilize
    before attempting detection.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_data = {}
        self.max_wait_time = config.get("max_wait_time", 5.0)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, None]:
        """Initialize wait tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_data[instance_id] = {
            "total_wait_time": 0.0,
            "wait_count": 0,
            "confidence_before": kwargs.get("confidence", 0.5)
        }
        return instance_id, None

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute wait operation."""
        if instance_id not in self._instance_data:
            return "Error: Invalid instance ID", -0.1, {}
        
        data = self._instance_data[instance_id]
        
        # Parse wait duration
        duration = min(parameters.get("duration", 1.0), self.max_wait_time - data["total_wait_time"])
        
        if duration <= 0:
            return f"Maximum wait time ({self.max_wait_time}s) exceeded", -0.2, {"wait_count": data["wait_count"]}
        
        # Update tracking
        data["total_wait_time"] += duration
        data["wait_count"] += 1
        
        # Simulate wait effect (in production, this would actually wait)
        # Confidence gain is higher for first wait, diminishes with subsequent waits
        base_gain = 0.15
        confidence_gain = base_gain * (0.7 ** (data["wait_count"] - 1))
        
        # Tool reward based on effectiveness
        if confidence_gain > 0.05 and data["wait_count"] <= 2:
            tool_reward = confidence_gain
        else:
            tool_reward = -0.1  # Penalty for excessive waiting
        
        return f"Waited {duration:.1f}s for UI to stabilize", tool_reward, {
            "confidence_gain": confidence_gain,
            "total_wait_time": data["total_wait_time"],
            "wait_count": data["wait_count"]
        }

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up instance data."""
        if instance_id in self._instance_data:
            del self._instance_data[instance_id]


class InspectTool(BaseTool):
    """Inspect tool for analyzing UI element structure and properties.
    
    This tool provides additional context about UI elements that may
    not be visible in the raw image.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_data = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> Tuple[str, None]:
        """Initialize inspect tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_data[instance_id] = {
            "inspect_count": 0,
            "confidence_before": kwargs.get("confidence", 0.5),
            "inspected_regions": []
        }
        return instance_id, None

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute element inspection."""
        if instance_id not in self._instance_data:
            return "Error: Invalid instance ID", -0.1, {}
        
        data = self._instance_data[instance_id]
        
        # Parse inspection target
        region = parameters.get("region", [0, 0, 1, 1])
        inspect_type = parameters.get("type", "structure")  # structure, properties, accessibility
        
        data["inspect_count"] += 1
        data["inspected_regions"].append(region)
        
        # Simulate inspection results (in production, this would analyze actual UI)
        if inspect_type == "structure":
            # Provide structural hints
            confidence_gain = 0.1
            info = "Detected nested container with 3 child elements"
        elif inspect_type == "properties":
            # Provide property information
            confidence_gain = 0.15
            info = "Element type: button, state: enabled, z-index: 1000"
        elif inspect_type == "accessibility":
            # Provide accessibility information
            confidence_gain = 0.12
            info = "Role: button, label: 'Submit Form', tabindex: 0"
        else:
            confidence_gain = 0.05
            info = "Unknown inspection type"
        
        # Diminishing returns for multiple inspections
        confidence_gain *= (0.8 ** (data["inspect_count"] - 1))
        
        # Tool reward
        if confidence_gain > 0.08:
            tool_reward = confidence_gain
        else:
            tool_reward = -0.05
        
        return f"Inspection result: {info}", tool_reward, {
            "confidence_gain": confidence_gain,
            "inspect_count": data["inspect_count"],
            "inspect_type": inspect_type
        }

    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up instance data."""
        if instance_id in self._instance_data:
            del self._instance_data[instance_id]