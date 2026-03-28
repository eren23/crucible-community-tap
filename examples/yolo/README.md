# YOLO Object Detection via MCP

Fine-tune YOLOv8/v11 on a cloud GPU using Crucible's MCP tools. No code to write — just config and tool calls.

## What Happened

We provisioned an RTX 4090 on RunPod, cloned Ultralytics, trained YOLOv8n on COCO8 for 10 epochs, collected mAP metrics, and tore down the pod. Total wall time: ~7 minutes. Total cost: ~$0.10.

**Results:**
```
precision_b:  0.679
recall_b:     0.750
mAP50_b:      0.772
mAP50-95_b:   0.606
```

## MCP Tool Sequence (8 calls)

```
list_projects          ->  found "yolo11-demo"
provision_project      ->  created RTX 4090 pod
fleet_refresh          ->  got SSH endpoint
bootstrap_project      ->  cloned Ultralytics, installed deps (~4 min)
run_project            ->  launched training (10 epochs)
get_fleet_status       ->  GPU at 41C, 15% disk
collect_project_results ->  parsed mAP from results.csv
destroy_nodes          ->  cleaned up
```

The full trace is in `trace.jsonl` — every tool call with timing and response.

## Try It Yourself

### 1. Copy the project spec

```bash
mkdir -p .crucible/projects
cp yolo11-demo.yaml .crucible/projects/
```

### 2. Set secrets

```bash
echo "RUNPOD_API_KEY=your_key" >> .env
echo "WANDB_API_KEY=your_key" >> .env
```

### 3. Run via MCP

```python
provision_project(project_name="yolo11-demo", count=1)
fleet_refresh()
bootstrap_project(project_name="yolo11-demo")
run_project(project_name="yolo11-demo", overrides={
    "MODEL": "yolo11n.pt",
    "DATA": "coco128.yaml",
    "EPOCHS": "30",
    "RUN_NAME": "my-yolo-run"
})
collect_project_results(run_id="...")
destroy_nodes()
```

### Swap models

| Model | Size | Speed |
|-------|------|-------|
| `yolov8n.pt` | Nano (3.2M) | Fastest |
| `yolov8s.pt` | Small (11.2M) | Fast |
| `yolo11n.pt` | v11 Nano (2.6M) | Fastest v11 |
| `yolo11s.pt` | v11 Small (9.4M) | Fast v11 |

### Swap datasets

| Dataset | Images | Description |
|---------|--------|-------------|
| `coco8.yaml` | 8 | Smoke test |
| `coco128.yaml` | 128 | Quick baseline |
| `coco.yaml` | 118K | Full COCO |
