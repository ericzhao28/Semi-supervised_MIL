def train(graph, model, saver, sess, data_generator, log_dir, restore_itr=0):
    """
    Train the model.
    """
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    TOTAL_ITERS = FLAGS.metatrain_iterations
    prelosses, postlosses = [], []
    save_dir = log_dir + '/model'
    train_writer = tf.summary.FileWriter(log_dir, graph)
    # actual training.
    if restore_itr == 0:
        training_range = range(TOTAL_ITERS)
    else:
        training_range = range(restore_itr+1, TOTAL_ITERS)
    for itr in training_range:
        state, tgt_mu = data_generator.generate_data_batch(itr)
        statea = state[:, :FLAGS.update_batch_size*FLAGS.T, :]
        stateb = state[:, FLAGS.update_batch_size*FLAGS.T:, :]
        actiona = tgt_mu[:, :FLAGS.update_batch_size*FLAGS.T, :]
        actionb = tgt_mu[:, FLAGS.update_batch_size*FLAGS.T:, :]
        feed_dict = {model.statea: statea,
                    model.stateb: stateb,
                    model.actiona: actiona,
                    model.actionb: actionb}
        input_tensors = [model.train_op]
        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend([model.train_summ_op, model.total_loss1, model.total_losses2[model.num_updates-1]])
        with graph.as_default():
            results = sess.run(input_tensors, feed_dict=feed_dict)

        if itr != 0 and itr % SUMMARY_INTERVAL == 0:
            prelosses.append(results[-2])
            train_writer.add_summary(results[-3], itr)
            postlosses.append(results[-1])

        if itr != 0 and itr % PRINT_INTERVAL == 0:
            print 'Iteration %d: average preloss is %.2f, average postloss is %.2f' % (itr, np.mean(prelosses), np.mean(postlosses))
            prelosses, postlosses = [], []

        if itr != 0 and itr % TEST_PRINT_INTERVAL == 0:
            if FLAGS.val_set_size > 0:
                input_tensors = [model.val_summ_op, model.val_total_loss1, model.val_total_losses2[model.num_updates-1]]
                val_state, val_act = data_generator.generate_data_batch(itr, train=False)
                statea = val_state[:, :FLAGS.update_batch_size*FLAGS.T, :]
                stateb = val_state[:, FLAGS.update_batch_size*FLAGS.T:, :]
                actiona = val_act[:, :FLAGS.update_batch_size*FLAGS.T, :]
                actionb = val_act[:, FLAGS.update_batch_size*FLAGS.T:, :]
                feed_dict = {model.statea: statea,
                            model.stateb: stateb,
                            model.actiona: actiona,
                            model.actionb: actionb}
                with graph.as_default():
                    results = sess.run(input_tensors, feed_dict=feed_dict)
                train_writer.add_summary(results[0], itr)
                print 'Test results: average preloss is %.2f, average postloss is %.2f' % (np.mean(results[1]), np.mean(results[2]))

        if itr != 0 and (itr % SAVE_INTERVAL == 0 or itr == training_range[-1]):
            print 'Saving model to: %s' % (save_dir + '_%d' % itr)
            with graph.as_default():
                saver.save(sess, save_dir + '_%d' % itr)

