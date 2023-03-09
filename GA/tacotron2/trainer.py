import tensorflow as tf

import csv
import pynvml
import sys
import os
import psutil
sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow_tts
from .tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import Seq2SeqBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy

import redis

class Tacotron2Trainer(Seq2SeqBasedTrainer):
    """Tacotron2 Trainer class based on Seq2SeqBasedTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_mixed_precision=False,
        creature_id=""
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_mixed_precision (bool): Use mixed precision or not.

        """
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_mixed_precision=is_mixed_precision,
        )
        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "stop_token_loss",
            "mel_loss_before",
            "mel_loss_after",
            "guided_attention_loss",
        ]
        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

        self.creature_id = creature_id

        pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
        self.redis_conn = redis.Redis(connection_pool=pool)

    def compile(self, model, optimizer):
        super().compile(model, optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def _train_step(self, batch):
        """Here we re-define _train_step because apply input_signature make
        the training progress slower on my experiment. Note that input_signature
        is apply on based_trainer by default.
        """
        if self._already_apply_input_signature is False:
            self.one_step_forward = tf.function(
                self._one_step_forward, experimental_relax_shapes=True
            )
            self.one_step_evaluate = tf.function(
                self._one_step_evaluate, experimental_relax_shapes=True
            )
            self.one_step_predict = tf.function(
                self._one_step_predict, experimental_relax_shapes=True
            )
            self._already_apply_input_signature = True

        # run one_step_forward
        self.one_step_forward(batch)

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _one_step_evaluate_per_replica(self, batch):
        """One step evaluate per GPU

        Tacotron-2 used teacher-forcing when training and evaluation.
        So we need pass `training=True` for inference step.

        """
        outputs = self._model(**batch, training=True)
        _, dict_metrics_losses = self.compute_per_example_losses(batch, outputs)

        self.update_eval_metrics(dict_metrics_losses)

    def _one_step_predict_per_replica(self, batch):
        """One step predict per GPU

        Tacotron-2 used teacher-forcing when training and evaluation.
        So we need pass `training=True` for inference step.

        """
        outputs = self._model(**batch, training=True)
        return outputs

    def compute_per_example_losses(self, batch, outputs):
        """Compute per example losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.

        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model

        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        (
            decoder_output,
            post_mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs

        mel_loss_before = calculate_3d_loss(
            batch["mel_gts"], decoder_output, loss_fn=self.mae
        )
        mel_loss_after = calculate_3d_loss(
            batch["mel_gts"], post_mel_outputs, loss_fn=self.mae
        )

        # calculate stop_loss
        max_mel_length = (
            tf.reduce_max(batch["mel_lengths"])
            if self.config["use_fixed_shapes"] is False
            else [self.config["max_mel_length"]]
        )
        stop_gts = tf.expand_dims(
            tf.range(tf.reduce_max(max_mel_length), dtype=tf.int32), 0
        )  # [1, max_len]
        stop_gts = tf.tile(
            stop_gts, [tf.shape(batch["mel_lengths"])[0], 1]
        )  # [B, max_len]
        stop_gts = tf.cast(
            tf.math.greater_equal(stop_gts, tf.expand_dims(batch["mel_lengths"], 1)),
            tf.float32,
        )

        stop_token_loss = calculate_2d_loss(
            stop_gts, stop_token_predictions, loss_fn=self.binary_crossentropy
        )

        # calculate guided attention loss.
        attention_masks = tf.cast(
            tf.math.not_equal(batch["g_attentions"], -1.0), tf.float32
        )
        loss_att = tf.reduce_sum(
            tf.abs(alignment_historys * batch["g_attentions"]) * attention_masks,
            axis=[1, 2],
        )
        loss_att /= tf.reduce_sum(attention_masks, axis=[1, 2])

        per_example_losses = (
            stop_token_loss + mel_loss_before + mel_loss_after + loss_att
        )

        dict_metrics_losses = {
            "stop_token_loss": stop_token_loss,
            "mel_loss_before": mel_loss_before,
            "mel_loss_after": mel_loss_after,
            "guided_attention_loss": loss_att,
        }

        return per_example_losses, dict_metrics_losses
    
    def save_loss_record(self):
        record_dict = {}
        l1, l2, l3 = psutil.getloadavg()
        CPU_use = (l3/os.cpu_count()) * 100

        print(CPU_use)

        msg=""
        for key in self.eval_metrics.keys():
            msg = msg + "\n"+ f"(Steps: {self.steps}) eval_{key} = {self.eval_metrics[key].result():.4f}."
            record_dict[key] = f"{self.eval_metrics[key].result():.4f}"
        
        print(msg)
        
        pynvml.nvmlInit()  # 必须先初始化
            #1.获取显卡驱动版本

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        #4.获取第i个显卡的显存大小,已使用,剩余
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(meminfo.total) #第二块显卡总的显存大小
        print(meminfo.used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
        gpu_mem_used = meminfo.used/meminfo.total
        # print(meminfo.free) #第二块显卡剩余显存大小
        cpu_usage = psutil.cpu_percent()
        #结束
        pynvml.nvmlShutdown()

        log_dirname = os.path.join(self.config["outdir"], "logs")
        if not os.path.exists(log_dirname):
            os.makedirs(log_dirname)
        filename = os.path.join(log_dirname,f"Step_{self.steps}.log")
        with open(filename,'w') as f:
            f.write(msg)
            f.write(f"\nMemory Usage: {psutil.virtual_memory().percent}\n")
            f.write(f"CPU Usage: {cpu_usage}\n")
            f.write(f"GPU Usage: {gpu_mem_used}\n")
        
        record_dict["memory_usage"] = psutil.virtual_memory().percent
        record_dict["cpu_usage"] = cpu_usage
        record_dict["gpu_usage"] = gpu_mem_used
        
        performance_dirname = os.path.join(self.config["outdir"], "performance")
        if not os.path.exists(performance_dirname):
            os.makedirs(performance_dirname)
        csv_filename =  os.path.join(performance_dirname,f"Step_{self.steps}.csv")

        with open(csv_filename,'w',newline="") as f:
            writer = csv.DictWriter(f,fieldnames=record_dict.keys())
            writer.writeheader()
            writer.writerow(record_dict)

        self.redis_conn.hmset(self.creature_id,record_dict)

        # overall_performance_filepath = os.path.join(self.config["outdir"],'..','overall_performance.csv')
        # record_dict["creature_id"] = str(self.creature_id)
        # record_dict["step"] = self.steps
        # with open(overall_performance_filepath,'a',newline="") as f:
        #     writer = csv.DictWriter(f,fieldnames=record_dict.keys())
        #     writer.writerow(record_dict)
    def generate_and_save_intermediate_result(self, batch):
        self.save_loss_record()
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        outputs = self.one_step_predict(batch)
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            alignment_historys = alignment_historys.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(
            zip(mel_gts, mels_before, mels_after, alignment_historys), 0
        ):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plot figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment_history, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()