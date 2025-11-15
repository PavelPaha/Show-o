import os
import tempfile
import logging
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MoEMLflowLogger:
    def __init__(self, mlflow_client: Optional[MlflowClient] = None, mlflow_run_id: Optional[str] = None):
        self._mlflow_client = mlflow_client
        self._mlflow_run_id = mlflow_run_id
    
    def get_client_and_run_id(self):
        """Получает MLflow client и run_id"""
        if self._mlflow_client is not None and self._mlflow_run_id is not None:
            return self._mlflow_client, self._mlflow_run_id
        
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
        if run_id is None:
            return None, None
        
        client = MlflowClient()
        return client, run_id
    
    def log_gate_metrics(self, layer_id: int, global_step: int, 
                        expert_counts: Dict[int, int], total_activations: int, 
                        gate_score_mean: float, gate_score_std: float):
        """Логирует метрики гейтов для overall"""
        client, run_id = self.get_client_and_run_id()
        if client is None or run_id is None:
            return
        
        layer_prefix = f"moe/layer_{layer_id}"
        
        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        client.log_metric(run_id, f"{layer_prefix}/expert_balance", expert_balance, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/total_activations", total_activations, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_mean", gate_score_mean, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_std", gate_score_std, step=global_step)
    
    def log_modality_gate_metrics(self, layer_id: int, global_step: int, 
                                  expert_counts: Dict[int, int], total_activations: int,
                                  gate_score_mean: float, gate_score_std: float, modality_name: str):
        client, run_id = self.get_client_and_run_id()
        if client is None or run_id is None:
            return
        
        layer_prefix = f"moe/layer_{layer_id}/{modality_name}"
        
        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        client.log_metric(run_id, f"{layer_prefix}/expert_balance", expert_balance, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/total_activations", total_activations, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_mean", gate_score_mean, step=global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_std", gate_score_std, step=global_step)
    
    def log_all_plots(self, layer_id: int, global_step: int,
                     overall_heatmap_bytes: bytes,
                     overall_histogram_bytes: bytes,
                     combined_plot_bytes: bytes,
                     domain_plot_bytes: Optional[bytes] = None,
                     all_domains_plot_bytes: Optional[bytes] = None,
                     domain_id: Optional[str] = None):
        """Логирует все графики в MLflow за один раз"""
        client, run_id = self.get_client_and_run_id()
        if client is None or run_id is None:
            return
        
        layer_prefix = f"moe/layer_{layer_id}" if layer_id is not None else "moe"
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Overall heatmap
            tmp_file = os.path.join(temp_dir, f"gate_distribution_heatmap_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_heatmap_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            logger.info(f"📊 Heatmap распределений (overall) отправлена в MLflow: {layer_prefix}/gate_distribution_heatmap_step_{global_step}.png")
            
            # Overall histogram
            tmp_file = os.path.join(temp_dir, f"expert_token_counts_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_histogram_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            logger.info(f"📊 Количество токенов по экспертам отправлено: {layer_prefix}/expert_token_counts_step_{global_step}.png")
            
            # Combined plot (text + image)
            tmp_file = os.path.join(temp_dir, f"gate_distribution_combined_text_image_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(combined_plot_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            logger.info(f"📊 Объединенный график (text + image) отправлен в MLflow: {layer_prefix}/gate_distribution_combined_text_image_step_{global_step}.png")
            
            # Domain plot
            if domain_plot_bytes and domain_id:
                tmp_file = os.path.join(temp_dir, f"gate_distribution_domain_{domain_id}_step_{global_step}.png")
                with open(tmp_file, 'wb') as f:
                    f.write(domain_plot_bytes)
                client.log_artifact(run_id, tmp_file, layer_prefix)
                logger.info(f"📊 График для домена {domain_id} отправлен в MLflow: {layer_prefix}/gate_distribution_domain_{domain_id}_step_{global_step}.png")
            
            tmp_file = os.path.join(temp_dir, f"gate_distribution_all_domains_step_{global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(all_domains_plot_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            logger.info(f"📊 Общий график всех доменов отправлен в MLflow: {layer_prefix}/gate_distribution_all_domains_step_{global_step}.png")
        
        except Exception as e:
            logger.error(f"❌ Ошибка логирования графиков: {e}")
        finally:
            # Очистка временных файлов
            for f in os.listdir(temp_dir):
                os.unlink(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
    
    def log_distribution_heatmap(self, layer_id: int, global_step: int, 
                                heatmap_bytes: bytes, modality_name: str = "overall"):
        """Логирует отдельный heatmap для модальности"""
        client, run_id = self.get_client_and_run_id()
        if client is None or run_id is None:
            return
        
        temp_dir = tempfile.mkdtemp()
        layer_prefix = f"moe/layer_{layer_id}" if layer_id is not None else "moe"
        suffix = f"_{modality_name}" if modality_name != "overall" else ""
        filename = f"gate_distribution_heatmap{suffix}_step_{global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)
        
        try:
            with open(tmp_file_path, 'wb') as f:
                f.write(heatmap_bytes)
            client.log_artifact(run_id, tmp_file_path, layer_prefix)
            logger.info(f"📊 Heatmap распределений ({modality_name}) отправлена в MLflow: {layer_prefix}/{filename}")
        except Exception as e:
            logger.error(f"❌ Ошибка логирования heatmap: {e}")
        finally:
                   os.unlink(tmp_file_path)
                   os.rmdir(temp_dir)
    
    def log_layer_expert_heatmap(self, global_step: int, heatmap_bytes: bytes):
        """Логирует heatmap активации экспертов по слоям"""
        client, run_id = self.get_client_and_run_id()
        if client is None or run_id is None:
            return
        
        temp_dir = tempfile.mkdtemp()
        filename = f"expert_activation_frequency_layers_step_{global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)
        
        try:
            with open(tmp_file_path, 'wb') as f:
                f.write(heatmap_bytes)
            client.log_artifact(run_id, tmp_file_path, "moe")
            logger.info(f"📊 Heatmap активации экспертов по слоям отправлена в MLflow: moe/{filename}")
        except Exception as e:
            logger.error(f"❌ Ошибка логирования heatmap слои×эксперты: {e}")
        finally:
            os.unlink(tmp_file_path)
            os.rmdir(temp_dir)
