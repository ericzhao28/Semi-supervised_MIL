def generate_test_demos(data_generator):
    if not FLAGS.use_noisy_demos:
        n_folders = len(data_generator.demos.keys())
        demos = data_generator.demos
    else:
        n_folders = len(data_generator.noisy_demos.keys())
        demos = data_generator.noisy_demos
    policy_demo_idx = [np.random.choice(n_demo, replace=False, size=FLAGS.test_update_batch_size) \
                        for n_demo in [demos[i]['demoX'].shape[0] for i in xrange(n_folders)]]
    selected_demoO, selected_demoX, selected_demoU = [], [], []
    for i in xrange(n_folders):
        selected_cond = np.array(demos[i]['demoConditions'])[np.arange(len(demos[i]['demoConditions'])) == policy_demo_idx[i]]
        Xs, Us, Os = [], [], []
        for idx in selected_cond:
            if FLAGS.use_noisy_demos:
                demo_gif_dir = data_generator.noisy_demo_gif_dir
            else:
                demo_gif_dir = data_generator.demo_gif_dir
            O = np.array(imageio.mimread(demo_gif_dir + data_generator.gif_prefix + '_%d/cond%d.samp0.gif' % (i, idx)))[:, :, :, :3]
            O = np.transpose(O, [0, 3, 2, 1]) # transpose to mujoco setting for images
            O = O.reshape(FLAGS.T, -1) / 255.0 # normalize
            Os.append(O)
        Xs.append(demos[i]['demoX'][np.arange(demos[i]['demoX'].shape[0]) == policy_demo_idx[i]].squeeze())
        Us.append(demos[i]['demoU'][np.arange(demos[i]['demoU'].shape[0]) == policy_demo_idx[i]].squeeze())
        selected_demoO.append(np.array(Os))
        selected_demoX.append(np.array(Xs))
        selected_demoU.append(np.array(Us))
    print "Finished collecting demos for testing"
    selected_demo = dict(selected_demoX=selected_demoX, selected_demoU=selected_demoU, selected_demoO=selected_demoO)
    data_generator.selected_demo = selected_demo


