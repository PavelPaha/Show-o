import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import torch


class MoEVisualizer:
    def __init__(self, num_experts: int, layer_id: Optional[int] = None):
        self.num_experts = num_experts
        self.layer_id = layer_id
        
        
    def _plot_expert_counts(
        self,
        ax,
        counts: Optional[Dict[int, int]],
        title: str,
        color: str,
    ):
        experts = list(range(self.num_experts))
        values = [counts.get(e, 0) if counts else 0 for e in experts]
        bars = ax.bar(
            experts,
            values,
            alpha=0.7,
            color=color,
            edgecolor="black",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Expert ID", fontsize=10)
        ax.set_ylabel("Number of Activations", fontsize=10)
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01 if max(values) > 0 else 0.5,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        total_activations = sum(values)
        balance = max(values) - min(values) if values else 0
        ax.text(
            0.02,
            0.98,
            f"Total: {total_activations}\nBalance: {balance}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    def create_distribution_heatmap(
        self,
        distribution_history: Dict[int, Dict[int, int]],
        modality_name: str = "overall",
        global_step: int = 0,
        alpha_value: Optional[float] = None,
    ) -> bytes:
        distribution_history = distribution_history[modality_name]
        steps = sorted(distribution_history.keys())
        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(expert_ids), len(steps)))

        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total

        fig, ax = plt.subplots(figsize=(14, 8))
        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution{title_suffix}"
        if modality_name != "overall":
            title = f"Gate Distribution ({modality_name}){title_suffix}"

        im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Expert ID", fontsize=12)

        step_indices = np.arange(len(steps))
        ax.set_xticks(step_indices)
        ax.set_xticklabels(steps, rotation=45)
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Gate Distribution", fontsize=11)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_layer_expert_activation_heatmap(
        self, layer_expert_counts: Dict[int, Dict[int, int]], global_step: int = 0
    ) -> bytes:
        layer_ids = sorted(layer_expert_counts.keys())
        all_expert_ids = set()
        for layer_id, expert_counts in layer_expert_counts.items():
            all_expert_ids.update(expert_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(layer_ids), len(expert_ids)))

        for row_idx, layer_id in enumerate(layer_ids):
            expert_counts = layer_expert_counts[layer_id]
            total_activations = sum(expert_counts.values())

            if total_activations > 0:
                for col_idx, expert_id in enumerate(expert_ids):
                    count = expert_counts.get(expert_id, 0)

                    matrix[row_idx, col_idx] = count / total_activations
            else:

                pass

        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_ids) * 0.5)))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Expert Activation Frequency - Step {global_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Expert Index", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)

        ax.set_xticks(np.arange(len(expert_ids)))
        ax.set_xticklabels(expert_ids)
        ax.set_yticks(np.arange(len(layer_ids)))
        ax.set_yticklabels(layer_ids)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation Frequency", fontsize=11)

        if len(layer_ids) <= 24 and len(expert_ids) <= 16:
            for row_idx in range(len(layer_ids)):
                for col_idx in range(len(expert_ids)):
                    value = matrix[row_idx, col_idx]
                    if value > 0.01:
                        text = ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value > 0.5 else "black",
                            fontsize=8,
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_single_heatmap(
        self,
        ax,
        distribution_history: Dict[int, Dict[int, int]],
        modality_name: str = "overall",
        alpha_value: Optional[float] = None,
    ) -> bool:

        if not distribution_history:
            return False

        steps = sorted(distribution_history.keys())
        if not steps:
            return False

        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)

        if len(expert_ids) == 0:
            return False

        matrix = np.zeros((len(expert_ids), len(steps)))
        has_data = False

        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                has_data = True
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total

        if not has_data or matrix.max() == 0:
            return False

        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution ({modality_name}){title_suffix}"

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Expert ID", fontsize=10)

        step_indices = np.arange(len(steps))
        ax.set_xticks(step_indices)
        ax.set_xticklabels(steps, rotation=45, fontsize=8)
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids, fontsize=8)

        plt.colorbar(im, ax=ax, label="Gate Distribution")
        return True

    def create_expert_activation_histogram(
        self,
        expert_counts: Dict[int, int],
        modality_name: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Optional[bytes]:

        experts = list(expert_counts.keys())
        activations = list(expert_counts.values())

        create_new_figure = ax is None
        if create_new_figure:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

        bars = ax.bar(experts, activations, alpha=0.7, color="green", edgecolor="black")
        title = f"Expert Token Counts"
        if modality_name:
            title += f" ({modality_name})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Expert ID", fontsize=10)
        ax.set_ylabel("Number of Activations", fontsize=10)
        ax.grid(True, alpha=0.3)

        for bar, count in zip(bars, activations):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        total_activations = sum(activations)
        balance = max(activations) - min(activations) if activations else 0
        ax.text(
            0.02,
            0.98,
            f"Total: {total_activations}\nBalance: {balance}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        if create_new_figure:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close()
            return buf.getvalue()

    def create_modality_combined_plot(
        self,
        text_history: Optional[Dict[int, Dict[int, int]]] = None,
        image_history: Optional[Dict[int, Dict[int, int]]] = None,
        text_expert_counts: Optional[Dict[int, int]] = None,
        image_expert_counts: Optional[Dict[int, int]] = None,
        global_step: int = 0,
        alpha_value: Optional[float] = None,
    ) -> bytes:

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(
            f"Gate Distribution Analysis - Layer {self.layer_id} - Step {global_step}",
            fontsize=16,
            fontweight="bold",
        )

        self.create_single_heatmap(
            axes[0, 0], text_history, "text", alpha_value
        )
        self.create_single_heatmap(
            axes[0, 1], image_history, "image", alpha_value
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def create_layer_expert_activation_heatmap(
        self, layer_expert_counts: Dict[int, Dict[int, int]], global_step: int = 0
    ) -> bytes:
        layer_ids = sorted(layer_expert_counts.keys())
        all_expert_ids = set()
        for layer_id, expert_counts in layer_expert_counts.items():
            all_expert_ids.update(expert_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(layer_ids), len(expert_ids)))

        for row_idx, layer_id in enumerate(layer_ids):
            expert_counts = layer_expert_counts[layer_id]
            total_activations = sum(expert_counts.values())

            if total_activations > 0:
                for col_idx, expert_id in enumerate(expert_ids):
                    count = expert_counts.get(expert_id, 0)

                    matrix[row_idx, col_idx] = count / total_activations
            else:

                pass

        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_ids) * 0.5)))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Expert Activation Frequency - Step {global_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Expert Index", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)

        ax.set_xticks(np.arange(len(expert_ids)))
        ax.set_xticklabels(expert_ids)
        ax.set_yticks(np.arange(len(layer_ids)))
        ax.set_yticklabels(layer_ids)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation Frequency", fontsize=11)

        if len(layer_ids) <= 24 and len(expert_ids) <= 16:
            for row_idx in range(len(layer_ids)):
                for col_idx in range(len(expert_ids)):
                    value = matrix[row_idx, col_idx]
                    if value > 0.01:
                        text = ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value > 0.5 else "black",
                            fontsize=8,
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_domain_plot(
        self,
        domain_id: str,
        domain_history: Dict[int, Dict[int, int]],
        global_step: int,
        current_expert_counts: Optional[Dict[int, int]] = None,
    ) -> bytes:

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(
            f"Gate Distribution Analysis - {domain_id} - Layer {self.layer_id} - Step {global_step}",
            fontsize=16,
            fontweight="bold",
        )

        self.create_single_heatmap(axes[0], domain_history, domain_id)  
        self._plot_expert_counts(
            axes[1],
            current_expert_counts,
            f"Expert Token Counts ({domain_id})",
            color="purple",
        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_layer_expert_activation_heatmap(
        self, layer_expert_counts: Dict[int, Dict[int, int]], global_step: int = 0
    ) -> bytes:
        layer_ids = sorted(layer_expert_counts.keys())
        all_expert_ids = set()
        for layer_id, expert_counts in layer_expert_counts.items():
            all_expert_ids.update(expert_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(layer_ids), len(expert_ids)))

        for row_idx, layer_id in enumerate(layer_ids):
            expert_counts = layer_expert_counts[layer_id]
            total_activations = sum(expert_counts.values())

            if total_activations > 0:
                for col_idx, expert_id in enumerate(expert_ids):
                    count = expert_counts.get(expert_id, 0)

                    matrix[row_idx, col_idx] = count / total_activations
            else:

                pass

        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_ids) * 0.5)))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Expert Activation Frequency - Step {global_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Expert Index", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)

        ax.set_xticks(np.arange(len(expert_ids)))
        ax.set_xticklabels(expert_ids)
        ax.set_yticks(np.arange(len(layer_ids)))
        ax.set_yticklabels(layer_ids)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation Frequency", fontsize=11)

        if len(layer_ids) <= 24 and len(expert_ids) <= 16:
            for row_idx in range(len(layer_ids)):
                for col_idx in range(len(expert_ids)):
                    value = matrix[row_idx, col_idx]
                    if value > 0.01:
                        text = ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value > 0.5 else "black",
                            fontsize=8,
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_all_domains_combined_plot(
        self,
        domain_distribution_history: Dict[str, Dict[int, Dict[int, int]]],
        global_step: int,
    ) -> bytes:
        # Filter out modality keys, keep only actual domains
        # modality_keys = {'overall', 'text', 'image', 'video'}
        all_domains = [k for k in domain_distribution_history.keys()]
        all_domains = sorted(all_domains)
        print(f'[Visualizer] All domains after filtering: {all_domains}')
        n_domains = len(all_domains)

        fig, axes = plt.subplots(n_domains, 2, figsize=(12, 4 * n_domains))
        fig.suptitle(
            f"Gate Distribution by Domain - Layer {self.layer_id} - Step {global_step}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if n_domains == 1:
            axes = axes.reshape(1, -1)

        for idx, domain_id in enumerate(all_domains):
            ax_heatmap = axes[idx, 0]
            ax_dist = axes[idx, 1]

            domain_history = domain_distribution_history.get(domain_id, {})
            steps = sorted(domain_history.keys())
            recent_steps = steps

            heatmap_data = np.zeros((self.num_experts, len(recent_steps)))

            for step_idx, step in enumerate(recent_steps):
                step_data = domain_history[step]
                for expert_id, count in step_data.items():
                    if expert_id < self.num_experts:
                        heatmap_data[expert_id, step_idx] = count

            if heatmap_data.sum() > 0:
                heatmap_data_norm = heatmap_data / (
                    heatmap_data.sum(axis=0, keepdims=True) + 1e-8
                )
            else:
                heatmap_data_norm = heatmap_data

            im = ax_heatmap.imshow(
                heatmap_data_norm,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax_heatmap.set_title(
                f"{domain_id} - Heatmap", fontsize=11, fontweight="bold"
            )
            ax_heatmap.set_xlabel("Step", fontsize=9)
            ax_heatmap.set_ylabel("Expert ID", fontsize=9)
            ax_heatmap.set_yticks(range(self.num_experts))
            ax_heatmap.set_yticklabels(range(self.num_experts))
            
            # Отображаем подписи шагов с интервалом, чтобы не перекрывались
            if len(recent_steps) > 0:
                # Показываем подписи с интервалом, чтобы было не более 20 меток
                step_interval = max(1, len(recent_steps) // 20)
                tick_indices = range(0, len(recent_steps), step_interval)
                tick_labels = [str(recent_steps[i]) for i in tick_indices]
                ax_heatmap.set_xticks(tick_indices)
                ax_heatmap.set_xticklabels(tick_labels, rotation=45, ha='right')
            plt.colorbar(
                im, ax=ax_heatmap, label="Normalized Gate Distribution"
            )
            
            current_expert_counts = None
            current_expert_counts = domain_history.get(global_step, {})
            expert_ids = list(current_expert_counts.keys())
            counts = list(current_expert_counts.values())
            bars = ax_dist.bar(
                expert_ids, counts, alpha=0.7, color="steelblue", edgecolor="black"
            )
            ax_dist.set_title(
                f"{domain_id} - Distribution (Step {global_step})",
                fontsize=11,
                fontweight="bold",
            )
            ax_dist.set_xlabel("Expert ID", fontsize=9)
            ax_dist.set_ylabel("Number of Activations", fontsize=9)
            ax_dist.grid(True, alpha=0.3)

            for bar, count in zip(bars, counts):
                if count > 0:
                    ax_dist.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(counts) * 0.01,
                        str(count),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def create_modality_probability_plot(
        self,
        probability_history: Dict[str, Dict[int, torch.Tensor]],
        global_step: int,
    ) -> bytes | None:
        keys = sorted(probability_history.keys())
        available = []
        for key in keys:
            history = probability_history.get(key, {})
            if history and any(value is not None for value in history.values()):
                available.append(key)
        if not available:
            return None

        fig, axes = plt.subplots(
            len(available),
            2,
            figsize=(14, max(3, 2.5 * len(available))),
            squeeze=False,
        )
        fig.suptitle(
            f"Gate Probability by Modality/Domain - Layer {self.layer_id} - Step {global_step}",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        for idx, key in enumerate(available):
            modality_history = probability_history.get(key, {})
            steps = sorted(modality_history.keys())
            recent_steps = steps[-min(10, len(steps)) :]
            ax_heatmap = axes[idx, 0]
            if recent_steps:
                heatmap_data = []
                for step in recent_steps:
                    probs_tensor = modality_history[step]
                    probs = (
                        probs_tensor.detach().cpu().float().numpy()
                        if isinstance(probs_tensor, torch.Tensor)
                        else np.array(probs_tensor)
                    )
                    heatmap_data.append(probs)
                heatmap_matrix = np.stack(heatmap_data, axis=1)
                im = ax_heatmap.imshow(
                    heatmap_matrix,
                    aspect="auto",
                    cmap="viridis",
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )
                ax_heatmap.set_title(f"{key} - history", fontsize=12, fontweight="bold")
                ax_heatmap.set_xlabel("Recent Steps", fontsize=10)
                ax_heatmap.set_ylabel("Expert ID", fontsize=10)
                ax_heatmap.set_yticks(range(self.num_experts))
                ax_heatmap.set_yticklabels(range(self.num_experts))
                ax_heatmap.set_xticks(range(len(recent_steps)))
                ax_heatmap.set_xticklabels([str(s) for s in recent_steps], rotation=45, ha="right")
                plt.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
            else:
                ax_heatmap.set_title(f"{key} - history", fontsize=12, fontweight="bold")
                ax_heatmap.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax_heatmap.transAxes,
                )
                ax_heatmap.axis("off")

            ax_bar = axes[idx, 1]
            probs_tensor = modality_history.get(global_step)
            if probs_tensor is None:
                ax_bar.set_title(f"{key} - avg probs (no data)", fontsize=12, fontweight="bold")
                ax_bar.axis("off")
                continue
            probs = (
                probs_tensor.detach().cpu().float().numpy()
                if isinstance(probs_tensor, torch.Tensor)
                else np.array(probs_tensor)
            )
            expert_ids = np.arange(len(probs))
            ax_bar.bar(expert_ids, probs, color="darkorange", edgecolor="black", alpha=0.8)
            max_val = probs.max() if probs.size else 1.0
            ax_bar.set_ylim(0.0, max(1.0, max_val * 1.1))
            ax_bar.set_title(f"{key} - avg probs", fontsize=12, fontweight="bold")
            ax_bar.set_xlabel("Expert ID", fontsize=10)
            ax_bar.set_ylabel("Probability", fontsize=10)
            ax_bar.grid(True, alpha=0.3)
            if probs.size:
                for expert_id, value in zip(expert_ids, probs):
                    if value > 0.01:
                        ax_bar.text(
                            expert_id,
                            value + 0.01,
                            f"{value:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf.getvalue()
    def create_layer_probability_heatmap(
        self,
        layer_probability_map: Dict[int, torch.Tensor],
        modality_name: str,
        global_step: int,
    ) -> bytes | None:
        if not layer_probability_map:
            return None

        layer_ids = sorted(layer_probability_map.keys())
        matrix = np.zeros((len(layer_ids), self.num_experts))
        for row_idx, layer_id in enumerate(layer_ids):
            probs_tensor = layer_probability_map[layer_id]
            probs = (
                probs_tensor.detach().cpu().float().numpy()
                if isinstance(probs_tensor, torch.Tensor)
                else np.array(probs_tensor)
            )
            matrix[row_idx, : len(probs)] = probs

        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_ids) * 0.5)))
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"{modality_name} Probability Heatmap - Step {global_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Expert ID", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)
        ax.set_xticks(range(self.num_experts))
        ax.set_xticklabels(range(self.num_experts))
        ax.set_yticks(range(len(layer_ids)))
        ax.set_yticklabels(layer_ids)
        plt.colorbar(im, ax=ax, label="Probability")

        if len(layer_ids) <= 24 and self.num_experts <= 16:
            for row_idx in range(len(layer_ids)):
                for col_idx in range(self.num_experts):
                    value = matrix[row_idx, col_idx]
                    if value > 0.01:
                        ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value > 0.5 else "black",
                            fontsize=8,
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()
        return buf.getvalue()

    def create_layer_expert_activation_heatmap(
        self, layer_expert_counts: Dict[int, Dict[int, int]], global_step: int = 0
    ) -> bytes:
        layer_ids = sorted(layer_expert_counts.keys())
        all_expert_ids = set()
        for layer_id, expert_counts in layer_expert_counts.items():
            all_expert_ids.update(expert_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(layer_ids), len(expert_ids)))

        for row_idx, layer_id in enumerate(layer_ids):
            expert_counts = layer_expert_counts[layer_id]
            total_activations = sum(expert_counts.values())

            if total_activations > 0:
                for col_idx, expert_id in enumerate(expert_ids):
                    count = expert_counts.get(expert_id, 0)

                    matrix[row_idx, col_idx] = count / total_activations
            else:
                pass

        fig, ax = plt.subplots(figsize=(10, max(8, len(layer_ids) * 0.5)))

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        ax.set_title(
            f"Expert Activation Frequency - Step {global_step}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Expert Index", fontsize=12)
        ax.set_ylabel("Layer Index", fontsize=12)

        ax.set_xticks(np.arange(len(expert_ids)))
        ax.set_xticklabels(expert_ids)
        ax.set_yticks(np.arange(len(layer_ids)))
        ax.set_yticklabels(layer_ids)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Activation Frequency", fontsize=11)

        if len(layer_ids) <= 24 and len(expert_ids) <= 16:
            for row_idx in range(len(layer_ids)):
                for col_idx in range(len(expert_ids)):
                    value = matrix[row_idx, col_idx]
                    if value > 0.01:
                        text = ax.text(
                            col_idx,
                            row_idx,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if value > 0.5 else "black",
                            fontsize=8,
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()
