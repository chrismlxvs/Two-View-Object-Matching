# Validation notes

Selected configuration: shared YOLO26n detector at 960-pixel input with a scene-level matcher. Exact runtime settings and validation metrics are recorded in `selected_configuration.json`.

| Metric | Validation result |
| --- | ---: |
| Detection F1 | 0.98051 |
| Association F1 | 0.95306 |
| Given-detection association F1 | 0.98402 |
| Count MAE | 0.230 |
| Exact-count accuracy | 0.790 |

These measurements are from the labeled validation split only. Test accuracy was not measured. Do not tune using test labels. See the report source in `report/` for the method, comparison experiments, and limitations.
