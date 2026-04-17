import keras.callbacks as kc


class callbacks_:
    def __init__(self):
        self.built = {"callbacks": []}

    def modelCheckPoint(
        self,
        filepath=None,
        monitor=None,
        verbose=None,
        save_best_only=None,
        save_weights_only=None,
        mode=None,
        save_freq=None,
        initial_value_threshold=None,
    ):
        layer = kc.ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            initial_value_threshold=initial_value_threshold,
        )
        self.built["callbacks"].append(layer)

    def backupAndRestore(
        self,
        backup_dir=None,
        save_freq=None,
        double_checkpoint=None,
        delete_checkpoint=None,
    ):
        layer = kc.BackupAndRestore(
            backup_dir=backup_dir,
            save_freq=save_freq,
            double_checkpoint=double_checkpoint,
            delete_checkpoint=delete_checkpoint,
        )
        self.built["callbacks"].append(layer)

    def tENSORbOARD(
        self,
        log_dir=None,
        histogram_freq=None,
        write_graph=None,
        write_images=None,
        write_steps_per_second=None,
        update_freq=None,
        profile_batch=None,
        embeddings_freq=None,
        embeddings_metadata=None,
    ):
        layer = kc.TensorBoard(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            write_images=write_images,
            write_steps_per_second=write_steps_per_second,
            update_freq=update_freq,
            profile_batch=profile_batch,
            embeddings_freq=embeddings_freq,
            embeddings_metadata=embeddings_metadata,
        )
        self.built["callbacks"].append(layer)

    def earlyStopping(
        self,
        monitor=None,
        min_delta=None,
        patience=None,
        verbose=None,
        mode=None,
        baseline=None,
        restore_best_weights=None,
        start_from_epoch=None,
    ):
        layer = kc.EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
            start_from_epoch=start_from_epoch,
        )
        self.built["callbacks"].append(layer)

    def lrScheduler(self, schedule=None, verbose=None):
        layer = kc.LearningRateScheduler(schedule=schedule, verbose=verbose)
        self.built["callbacks"].append(layer)

    def reduceLRonPlateau(
        self,
        monitor=None,
        factor=None,
        patience=None,
        verbose=None,
        mode=None,
        min_delta=None,
        cooldown=None,
        min_lr=None,
        **kwargs
    ):
        layer = kc.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs
        )
        self.built["callbacks"].append(layer)

    def remoteMonitor(
        self,
        root=None,
        path=None,
        field=None,
        headers=None,
        send_as_json=None,
    ):
        layer = kc.RemoteMonitor(
            root=root,
            path=path,
            field=field,
            headers=headers,
            send_as_json=send_as_json,
        )
        self.built["callbacks"].append(layer)

    def lambdaCallback(
        self,
        on_epoch_begin=None,
        on_epoch_end=None,
        on_train_begin=None,
        on_train_end=None,
        on_train_batch_begin=None,
        on_train_batch_end=None,
        **kwargs
    ):
        layer = kc.LambdaCallback(
            on_epoch_begin=on_epoch_begin,
            on_epoch_end=on_epoch_end,
            on_train_begin=on_train_begin,
            on_train_end=on_train_end,
            on_train_batch_begin=on_train_batch_begin,
            on_train_batch_end=on_train_batch_end,
            **kwargs
        )
        self.built["callbacks"].append(layer)

    def terminateON_NaN(self, raise_error: bool = None):
        layer = kc.TerminateOnNaN(raise_error=raise_error)
        self.built["callbacks"].append(layer)

    def CsvLogger(self, filename=None, separator=None, append=None):
        layer = kc.CSVLogger(filename=filename, separator=separator, append=append)
        self.built["callbacks"].append(layer)

    def progBarLogger(self):
        layer = kc.ProgbarLogger()
        self.built["callbacks"].append(layer)

    def swapEMAWeights(self, swap_on_epoch=None):
        layer = kc.SwapEMAWeights(swap_on_epoch=swap_on_epoch)
        self.built["callbacks"].append(layer)

    def orbaxCheckpoint(
        self,
        directory=None,
        monitor=None,
        verbose=None,
        save_best_only=None,
        mode=None,
        save_freq=None,
        initial_value_threshold=None,
        max_to_keep=None,
        save_on_background=None,
        save_weights_only=None,
    ):
        layer = kc.OrbaxCheckpoint(
            directory=directory,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            mode=mode,
            save_freq=save_freq,
            initial_value_threshold=initial_value_threshold,
            max_to_keep=max_to_keep,
            save_on_background=save_on_background,
            save_weights_only=save_weights_only,
        )
        self.built["callbacks"].append(layer)
