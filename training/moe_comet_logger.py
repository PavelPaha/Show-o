import os
import tempfile
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MoECometLogger:
    def __init__(self, comet_experiment=None):
        self._experiment = comet_experiment

    @property
    def experiment(self):
        return self._experiment

    def log_gate_metrics(self, layer_id: int, global_step: int,
                         expert_counts: Dict[int, int], total_activations: int,
                         gate_score_mean: float, gate_score_std: float):
        if self._experiment is None:
            return

        layer_prefix = f"moe/layer_{layer_id}"

        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        self._experiment.log_metric(f"{layer_prefix}/expert_balance", expert_balance, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/total_activations", total_activations, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/gate_weights_mean", gate_score_mean, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/gate_weights_std", gate_score_std, step=global_step)

    def log_modality_gate_metrics(self, layer_id: int, global_step: int,
                                  expert_counts: Dict[int, int], total_activations: int,
                                  gate_score_mean: float, gate_score_std: float, modality_name: str):
        if self._experiment is None:
            return

        layer_prefix = f"moe/layer_{layer_id}/{modality_name}"

        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        self._experiment.log_metric(f"{layer_prefix}/expert_balance", expert_balance, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/total_activations", total_activations, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/gate_weights_mean", gate_score_mean, step=global_step)
        self._experiment.log_metric(f"{layer_prefix}/gate_weights_std", gate_score_std, step=global_step)

    def log_all_plots(self, layer_id: int, global_step: int,
                      overall_heatmap_bytes: bytes,
                      overall_histogram_bytes: bytes,
                      combined_plot_bytes: bytes,
                      domain_plot_bytes: Optional[bytes] = None,
                      all_domains_plot_bytes: Optional[bytes] = None,
                      modality_probability_plot_bytes: Optional[bytes] = None,
                      domain_id: Optional[str] = None):
        if self._experiment is None:
            return

        layer_prefix = f"moe/layer_{layer_id}" if layer_id is not None else "moe"
        temp_dir = tempfile.mkdtemp()

        try:
            tmp_file = os.path.join(temp_dir, f"gate_distribution_heatmap_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_heatmap_bytes)
            self._experiment.log_image(tmp_file, name=f"{layer_prefix}/gate_distribution_heatmap_step_{global_step}", step=global_step)

            tmp_file = os.path.join(temp_dir, f"expert_token_counts_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_histogram_bytes)
            self._experiment.log_image(tmp_file, name=f"{layer_prefix}/expert_token_counts_step_{global_step}", step=global_step)

            tmp_file = os.path.join(temp_dir, f"gate_distribution_combined_text_image_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(combined_plot_bytes)
            self._experiment.log_image(tmp_file, name=f"{layer_prefix}/gate_distribution_combined_text_image_step_{global_step}", step=global_step)

            if domain_plot_bytes and domain_id:
                tmp_file = os.path.join(temp_dir, f"gate_distribution_domain_{domain_id}_step_{global_step}.png")
                with open(tmp_file, 'wb') as f:
                    f.write(domain_plot_bytes)
                self._experiment.log_image(tmp_file, name=f"{layer_prefix}/gate_distribution_domain_{domain_id}_step_{global_step}", step=global_step)

            if all_domains_plot_bytes is not None:
                tmp_file = os.path.join(temp_dir, f"gate_distribution_all_domains_step_{global_step}.png")
                with open(tmp_file, 'wb') as f:
                    f.write(all_domains_plot_bytes)
                self._experiment.log_image(tmp_file, name=f"{layer_prefix}/gate_distribution_all_domains_step_{global_step}", step=global_step)

            if modality_probability_plot_bytes is not None:
                tmp_file = os.path.join(temp_dir, f"gate_distribution_modalities_probs_step_{global_step}.png")
                with open(tmp_file, 'wb') as f:
                    f.write(modality_probability_plot_bytes)
                self._experiment.log_image(tmp_file, name=f"{layer_prefix}/gate_distribution_modalities_probs_step_{global_step}", step=global_step)

        except Exception as e:
            logger.error(f"Error logging plots to Comet: {e}")
        finally:
            for f_name in os.listdir(temp_dir):
                os.unlink(os.path.join(temp_dir, f_name))
            os.rmdir(temp_dir)

    def log_distribution_heatmap(self, layer_id: int, global_step: int,
                                 heatmap_bytes: bytes, modality_name: str = "overall"):
        if self._experiment is None:
            return

        temp_dir = tempfile.mkdtemp()
        layer_prefix = f"moe/layer_{layer_id}" if layer_id is not None else "moe"
        suffix = f"_{modality_name}" if modality_name != "overall" else ""
        filename = f"gate_distribution_heatmap{suffix}_step_{global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)

        try:
            with open(tmp_file_path, 'wb') as f:
                f.write(heatmap_bytes)
            self._experiment.log_image(tmp_file_path, name=f"{layer_prefix}/{filename}", step=global_step)
        except Exception as e:
            logger.error(f"Error logging heatmap to Comet: {e}")
        finally:
            os.unlink(tmp_file_path)
            os.rmdir(temp_dir)

    def log_layer_expert_heatmap(self, global_step: int, heatmap_bytes: bytes, suffix='all_tokens'):
        if self._experiment is None:
            return

        temp_dir = tempfile.mkdtemp()
        filename = f"expert_activation_frequency_layers_step_{suffix}_{global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)

        try:
            with open(tmp_file_path, 'wb') as f:
                f.write(heatmap_bytes)
            self._experiment.log_image(tmp_file_path, name=f"moe/{filename}", step=global_step)
        except Exception as e:
            logger.error(f"Error logging layer expert heatmap to Comet: {e}")
        finally:
            os.unlink(tmp_file_path)
            os.rmdir(temp_dir)

    def log_layer_probability_heatmap(self, global_step: int, heatmap_bytes: bytes, suffix='overall'):
        if self._experiment is None:
            return

        temp_dir = tempfile.mkdtemp()
        filename = f"expert_activation_prob_layers_step_{suffix}_{global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)

        try:
            with open(tmp_file_path, 'wb') as f:
                f.write(heatmap_bytes)
            self._experiment.log_image(tmp_file_path, name=f"moe/{filename}", step=global_step)
        except Exception as e:
            logger.error(f"Error logging probability heatmap to Comet: {e}")
        finally:
            os.unlink(tmp_file_path)
            os.rmdir(temp_dir)
