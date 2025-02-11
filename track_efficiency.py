from torch_geometric.data import Data, Dataset
from datasets.particle import TrackMLParticleTrackingDataset
from models.EdgeNetWithCategories import EdgeNetWithCategories
from models.InteractionNetwork import InteractionNetwork
# from princeton_gnn_tracking.models.EC1.ec1 import GNNSegmentClassifier

import pickle
import copy
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from unionfind import unionfind
from matplotlib import pyplot as plt


def main(args):
    """Main function"""
    # Parse the command line
    # args = parse_args()

    group = args.group
    model_group = args.model
    type = args.type
    config_file = '/data/gnn_code/training_data/' + group + '/config.yaml'
    # Nevents = args.n_events
    # pt_range = [args.pt_min, args.pt_max]

    with open(config_file) as f:
        config = yaml.load(f)
        selection = config['selection']
        # n_events = config['n_files']
        n_events = 200
        # n_events = 500
        # n_events = 10
        # n_events = 24



    trackml_data = TrackMLParticleTrackingDataset('/data/gnn_code/training_data/' + group,
    # trackml_data = TrackMLParticleTrackingDataset('/data/gnn_code/training_data/test_geometric',
                                                  volume_layer_ids=selection['volume_layer_ids'],
                                                  layer_pairs=selection['layer_pairs'],
                                                  layer_pairs_plus=selection['layer_pairs_plus'],
                                                  pt_range=selection['pt_range'],
                                                  # pt_range=pt_range,
                                                  eta_range=selection['eta_range'],
                                                  phi_slope_max=selection['phi_slope_max'],
                                                  # phi_slope_max=.001,
                                                  z0_max=selection['z0_max'],
                                                  diff_phi=selection['diff_phi'],
                                                  diff_z=selection['diff_z'],
                                                  # z0_max=150,
                                                  n_phi_sections=selection['n_phi_sections'],
                                                  n_eta_sections=selection['n_eta_sections'],
                                                  # n_phi_sections=4,
                                                  # n_eta_sections=2,
                                                  # augments=selection['construct_augmented_graphs'],
                                                  augments=False,
                                                  intersect=selection['remove_intersecting_edges'],
                                                  hough=selection['hough_transform'],
                                                  noise=selection['noise'],
                                                  duplicates=selection['duplicates'],
                                                  secondaries=selection['secondaries'],
                                                  tracking=True,
                                                  # n_workers=24,
                                                  n_tasks=1,
                                                  n_events=n_events,
                                                  data_type=config['data_type'],
                                                  mmap=selection['module_map'],
                                                  N_modules=selection['N_modules']
                                                  # n_events=Nevents,
                                                  # directed=True,
                                                  # layer_pairs_plus=True,
                                                  )

    if (args.reprocess):
        trackml_data.process(True)


    # Print properties of the data set
    print()
    print("Graph Construction Features")
    print("number_of_graphs = " + str(n_events))
    print("number_of_phi_sections = " + str(selection['n_phi_sections']))
    print("number_of_eta_sections = " + str(selection['n_eta_sections']))
    print("edge_features_used = " + str(selection['hough_transform']))
    print("layer_pairs_plus_used = " + str(selection['layer_pairs_plus']))
    print()
    print("Truth Level Cuts Applied")
    print("pt_range = " + str(selection['pt_range']))
    print("noise_hits_present = " + str(selection['noise']))
    print("duplicate_hits_from_same_paticle_within_layer_present = " + str(selection['duplicates']))
    print()
    print("Geometric Cuts Applied")
    print("eta_range = " + str(selection['eta_range']))
    print("phi_slope_max = " + str(selection['phi_slope_max']))
    print("z0_max = " + str(selection['z0_max']))
    print("diff_phi = " + str(selection['diff_phi']))
    print("diff_z = " + str(selection['diff_z']))
    print("intersecting_line_cut_used = " + str(selection['remove_intersecting_edges']))
    print()
    print("Inference Graph Node Properties")
    print("Average_Graph_Node_Count = " + str(trackml_data.average_node_count))
    print("Average_Total_Pixel_Node_Count = " + str(trackml_data.average_total_pixel_node_count))
    print("Average_Total_Global_Node_Count = " + str(trackml_data.average_total_node_count))
    print()
    print("Inference Graph Edge Properties")
    print("Average_Graph_Edge_Count = " + str(trackml_data.average_edge_count))
    print("Average_Graph_True_Edge_Count = " + str(trackml_data.average_true_edge_count))
    print("Average_Graph_Purity = " + str(trackml_data.average_true_edge_count / trackml_data.average_edge_count))
    print("Average_Pruned_Pixel_True_Edge_Count = " + str(trackml_data.average_pruned_pixel_true_edge_count))
    print("Average_Total_Pixel_True_Edge_Count = " + str(trackml_data.average_total_pixel_true_edge_count))
    print("Average_Total_Global_True_Edge_Count = " + str(trackml_data.average_total_true_edge_count))
    print()
    print("Inference Graph Track Properties")
    print("Average_Pixel_Track_Threshold_Count = " + str(trackml_data.average_pixel_track_threshold_count))
    print("Average_Pixel_Track_Count = " + str(trackml_data.average_pixel_track_count))
    print("Average_Global_Track_Count = " + str(trackml_data.average_total_track_count))
    print()

    # trackml_data.draw(0)


    # print(config_file)
    N_attr = trackml_data[0].edge_attr.shape[1]
    Hits_minimum = 3
    pt_cut_for_eta = 1.0


    if (type == "edge_classifier"):
        if(N_attr == 4):
            model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_EdgeNetWithCategories_259587_d1ebe35288_atkinsn2.best.pth.tar'
        else:
            model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_EdgeNetWithCategories_259075_140efb4178_atkinsn2.best.pth.tar'
        mdl = EdgeNetWithCategories(input_dim=3, hidden_dim=64, edge_dim=N_attr, output_dim=2, n_iters=6).to('cuda:0')
        # mdl = EdgeNetWithCategories(input_dim=3, hidden_dim=64, edge_dim=N_attr, output_dim=2, n_iters=6).to('cpu:0')
    elif (type == "interaction_network"):
        # model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_6448_77ce67e079_atkinsn2.best.pth.tar'
        # model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_9558_f9fcff2bfe_atkinsn2.best.pth.tar'
        # model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_22488_7d94caeb8a_atkinsn2.best.pth.tar'
        # model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_73658_99bfbda303_atkinsn2.best.pth.tar'
        # model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_282308_620b37ae37_atkinsn2.best.pth.tar'
        model_fname = '/data/gnn_code/hgcal_ldrd/output/'+model_group+'/checkpoints/model_checkpoint_InteractionNetwork_16721_5e48b5fd7c_atkinsn2.best.pth.tar'
        mdl = InteractionNetwork(input_dim=3, hidden_dim=40, edge_dim=N_attr, output_dim=2, n_iters=6).to('cuda:0')


    mdl.load_state_dict(torch.load(model_fname)['model'])
    mdl.eval()

    # Initializing vectors
    truth_pt    = []
    truth_eta   = []
    tru_pt      = []
    tru_eta     = []
    gnn_pt      = []
    gnn_eta     = []
    d_tru_pt    = []
    d_tru_eta   = []
    d_gnn_pt    = []
    d_gnn_eta   = []
    u_gnn_pt    = []
    u_gnn_eta   = []
    n_gnn_pt    = []
    n_gnn_eta   = []

    recon_pt    = []
    recon_eta   = []
    m_exa_pt    = []
    m_exa_eta   = []
    r_exa_pt    = []
    r_exa_eta   = []

    pred_edges_all = []
    y_all = []

    # Loop over number of events to average
    for idx in tqdm(range(n_events)):

        # Perform Inference
        with torch.no_grad():
            pred_edges = mdl(trackml_data[idx].to('cuda:0')).detach()
            # pred_edges = mdl(trackml_data[idx].to('cpu:0')).detach()
            pred_edges_np = pred_edges.cpu().numpy()

        if (type == "edge_classifier"):
            pred_edges_np = -pred_edges_np / (2* np.amin(pred_edges_np))
            pred_edges_np = pred_edges_np[:,1] - pred_edges_np[:,0] + .5
            # out = np.greater(pred_edges_np, 0.5).astype(np.int_)
            out = np.greater(pred_edges_np, 0.511).astype(np.int_)   #Triplets TrackML
            # out = np.greater(pred_edges_np, 0.522).astype(np.int_) #Triplets
            # out = np.greater(pred_edges_np, 0.555).astype(np.int_) #Triplets
            # out = np.greater(pred_edges_np, 0.5195).astype(np.int_) #Doublets

        elif (type == "interaction_network"):
            # out = np.transpose(np.greater(pred_edges_np, 0.5).astype(np.int_))[0]
            # out = np.transpose(np.greater(pred_edges_np, 0.1).astype(np.int_))[0]
            out = np.transpose(np.greater(pred_edges_np, 0.96).astype(np.int_))[0]
            pred_edges_np = np.transpose(pred_edges_np)[0]

        # Extract Graph features
        X = trackml_data[idx].x.cpu().numpy()
        index = trackml_data[idx].edge_index.cpu().numpy().T
        y = trackml_data[idx].y.cpu().numpy()
        particles = trackml_data[idx].particles.cpu().numpy()
        track_attr = trackml_data[idx].track_attr.cpu().numpy()

        pred_edges_all.append(pred_edges_np)
        y_all.append(y)

        if trackml_data.directed:
            f_pred_true_index = index[out == 1]
            f_pred_true_weights = pred_edges_np[out == 1]
        else:
            f_out, b_out = np.split(out, 2, axis=0)
            f_index, b_index = np.split(index, 2, axis=0)
            f_weight, b_weight = np.split(pred_edges_np, 2, axis=0)
            f_pred_true_index = f_index[f_out == 1]
            f_pred_true_weights = f_weight[f_out == 1]
            b_pred_true_index = b_index[b_out == 1]
            b_pred_true_weights = b_weight[b_out == 1]


        if (trackml_data.data_type == 'ATLAS'):
            track_attr = track_attr[track_attr[:,3] < 200000] #Prunes Secondaries from Denominator
            track_attr = np.delete(track_attr, 3, 1)

        track_prune = track_attr[track_attr[:,3] >= Hits_minimum] # Prunes Denominator
        # track_prune = track_attr # Full Denominator

        # track_prune =


        # Truth particle parameters
        truth_pt.append(track_prune[:,1])
        track_prune_eta = track_prune[track_prune[:,1] >= pt_cut_for_eta]
        truth_eta.append(track_prune_eta[:,2])

        # ExaTrk defines Reconstructable as particles with atleast 5 hits
        track_recon = track_attr[track_attr[:,3] >= 4]
        recon_pt.append(track_recon[:,1])
        track_recon_eta = track_recon[track_recon[:,1] >= pt_cut_for_eta]
        recon_eta.append(track_recon_eta[:,2])


        # Union Find the True edges
        finder_tru = unionfind(X.shape[0])
        for i in range(index.shape[0]):
            if y[i] == 1:
                finder_tru.unite(index[i,0], index[i,1])
        tru_roots = np.array([finder_tru.find(i) for i in range(X.shape[0])], dtype=np.uint32)
        tru_clusters = np.unique(tru_roots, return_inverse=True, return_counts=True)
        trus = tru_clusters[0][np.where(tru_clusters[2] >= Hits_minimum)]
        tru_clusters_sel = {i: np.where(tru_roots == tru)[0] for i, tru in enumerate(trus)}

        # True Graph particle parameters
        found_particles = np.empty(0, dtype=np.float64)
        for i in range(len(tru_clusters_sel)):
            track_particles = [particles[j] for j in tru_clusters_sel[i]]
            match = track_particles.count(track_particles[0]) == len(track_particles)
            found_previously = np.isin(track_particles[0], found_particles)
            if(match and not found_previously):
                track_pt, track_eta  = get_particle_pt_eta(track_particles[0], track_attr)
                tru_pt.append(track_pt)
                if(track_pt > pt_cut_for_eta):
                    tru_eta.append(track_eta)
                found_particles = np.append(found_particles, track_particles[0])
            elif(match and found_previously):
                track_pt, track_eta  = get_particle_pt_eta(track_particles[0], track_attr)
                d_tru_pt.append(track_pt)
                if(track_pt > pt_cut_for_eta):
                    d_tru_eta.append(track_eta)


        # Union Find the Inferenced edges
        finder_gnn = unionfind(X.shape[0])
        for i in range(index.shape[0]):
            if out[i] == 1:
                finder_gnn.unite(index[i,0], index[i,1])
        gnn_roots = np.array([finder_gnn.find(i) for i in range(X.shape[0])], dtype=np.uint32)
        gnn_clusters = np.unique(gnn_roots, return_inverse=True, return_counts=True)
        gnns = gnn_clusters[0][np.where(gnn_clusters[2] >= Hits_minimum)]
        gnn_clusters_sel = {i: np.where(gnn_roots == gnn)[0] for i, gnn in enumerate(gnns)}


        # Inferenced particle parameters
        found_particles     = np.empty(0, dtype=np.float64)
        found_particles_exa_p = np.empty(0, dtype=np.float64)
        found_particles_exa_t = np.empty(0, dtype=np.float64)

        for i in range(len(gnn_clusters_sel)):
            all_paths = [gnn_clusters_sel[i]]

            ########################  WRANGLER  ###########################
            # if trackml_data.directed:
            #     new_paths = find_all_paths(gnn_clusters_sel[i], f_pred_true_index, f_pred_true_weights)
            # else:
            #     # new_paths = find_all_paths(list(gnn_clusters_sel[i]), f_pred_true_index, f_pred_true_weights)
            #
            #     # new_paths = find_all_paths(list(gnn_clusters_sel[i])[::-1], b_pred_true_index, b_pred_true_weights)
            #
            #     f_paths = find_all_paths(list(gnn_clusters_sel[i]), f_pred_true_index, f_pred_true_weights)
            #     b_paths = find_all_paths(list(gnn_clusters_sel[i])[::-1], b_pred_true_index, b_pred_true_weights)
            #     b_paths = [path[::-1] for path in b_paths]
            #     new_paths = []
            #     for i in f_paths + b_paths:
            #         if i not in new_paths:
            #             new_paths.append(i)
            # if len(new_paths) > 0:
            #     new_paths = [new_paths[np.argmax(np.asarray([len(i) for i in new_paths]))]]             # Longest Path
            #     if len(new_paths[0]) >= Hits_minimum:
            #         all_paths = new_paths
            ########################  WRANGLER  ###########################


            for track_candidate in all_paths:
                track_particles = [particles[j] for j in track_candidate]

                #Markus strict matching criteria
                particle, count = np.unique(np.array(track_particles), return_counts=True)
                purity = np.max(count) / np.sum(count)
                particle = particle[np.argmax(count)]

                threshold = 1.0
                # threshold = 0.5

                found_previously = np.isin(particle, found_particles)
                if(purity >= threshold and particle == 0):  # Noise Track
                    track_pos = [X[j] for j in gnn_clusters_sel[i]]
                    track_pt = 0
                    track_eta = np.asarray([get_node_eta(track_pos[i]) for i in range(len(track_pos))])
                    n_gnn_pt.append(track_pt)
                    if(track_pt > pt_cut_for_eta):
                        n_gnn_eta.append(track_eta.mean())
                elif(purity >= threshold and not found_previously):   # Good Track
                    track_pt, track_eta  = get_particle_pt_eta(particle, track_attr)
                    gnn_pt.append(track_pt)
                    if(track_pt > pt_cut_for_eta):
                        gnn_eta.append(track_eta)
                    found_particles = np.append(found_particles, particle)
                elif(purity >= threshold and found_previously):       # Duplicate Track
                    track_pt, track_eta  = get_particle_pt_eta(particle, track_attr)
                    d_gnn_pt.append(track_pt)
                    if(track_pt > pt_cut_for_eta):
                        d_gnn_eta.append(track_eta)
                else:                                   # Bad Mixed Track
                    track_particles = [x for x in track_particles if x > 0] #prunes out noise hits
                    track_pt  = np.asarray([get_particle_pt_eta(track_particles[i], track_attr)[0] for i in range(len(track_particles))])
                    track_eta = np.asarray([get_particle_pt_eta(track_particles[i], track_attr)[1] for i in range(len(track_particles))])
                    u_gnn_pt.append(track_pt.mean())
                    if(track_pt.mean() > pt_cut_for_eta):
                        u_gnn_eta.append(track_eta.mean())


                #ExaTrk looser matching criteria
                particle, count = np.unique(np.array(track_particles), return_counts=True)
                ratio = np.max(count) / np.sum(count)
                index = np.argmax(count)
                count = np.max(count)
                particle = particle[index]
                found_previously_exa_p = np.isin(particle, found_particles_exa_p)
                found_previously_exa_t = np.isin(particle, found_particles_exa_t)
                #Require majority of hits from one particle
                if (ratio > .5):
                    index = np.where(track_attr[:,0] == particle)
                    ratio = count / track_attr[index,3]
                    if (ratio > .2 and not found_previously_exa_p): #Reduced for inner detector only
                        found_particles_exa_p = np.append(found_particles_exa_p, particle)
                        track_pt, track_eta   = get_particle_pt_eta(particle, track_attr)
                        m_exa_pt.append(track_pt)
                        if(track_pt > pt_cut_for_eta):
                            m_exa_eta.append(track_eta)

                    index = np.where(track_recon[:,0] == particle)
                    ratio = count / track_recon[index,3]
                    if (ratio > .2 and not found_previously_exa_t): #Reduced for inner detector only
                        found_particles_exa_t = np.append(found_particles_exa_t, particle)
                        track_pt, track_eta   = get_particle_pt_eta(particle, track_recon)
                        r_exa_pt.append(track_pt)
                        if(track_pt > pt_cut_for_eta):
                            r_exa_eta.append(track_eta)

    # # This chunk of code finds the optimal edge weight values to cut on
    # pred_edges_all = np.concatenate(pred_edges_all, axis=0)
    # y_all = np.concatenate(y_all, axis=0)
    #
    # fp = []
    # tn = []
    # xc = []
    # for i in tqdm(range(1001)):
    #     out_new = np.greater(pred_edges_all, i*.001).astype(np.int_)
    #     errors = out_new - y_all
    #     xc.append(i*.001)
    #     fp.append(np.count_nonzero(errors == 1))
    #     tn.append(np.count_nonzero(errors == -1))
    # N = y_all.shape[0]
    # true_N = np.sum(y_all)
    # xc = np.asarray(xc)
    # fp = np.asarray(fp)/(N-true_N)
    # tn = np.asarray(tn)/true_N
    # tot = fp+tn
    # print(xc[np.argmin(tot)])
    #
    # figg, (axg) = plt.subplots(1, 1, dpi=600, figsize=(6, 6))
    # axg.scatter(xc, fp, s=1, label="False Positives")
    # axg.scatter(xc, tn, s=1, label="True Negatives")
    # axg.set_xlabel('cut value')
    # axg.set_ylabel('percentage')
    # axg.set_ylim(.00000001, 1)
    # axg.set_yscale('log')
    # axg.legend()#loc='upper right')
    # figg.savefig('pos_neg.png', dpi=600)
    #
    # false_weights = pred_edges_all[y_all == 0]
    # true_weights  = pred_edges_all[y_all == 1]
    #
    # figh, (axh) = plt.subplots(1, 1, dpi=600, figsize=(6, 6))
    # axh.hist(false_weights, bins=100, range=(0,1), histtype='step', edgecolor='blue',   fill=False, label='False Edges')
    # axh.hist(true_weights, bins=100, range=(0,1), histtype='step', edgecolor='orange',   fill=False, label='True Edges')
    # axh.set_xlabel('edge score')
    # axh.set_ylabel('count')
    # # axh.set_ylim(.00000001, 1)
    # axh.set_yscale('log')
    # axh.legend()#loc='upper right')
    # figh.savefig('weights_distro.png', dpi=600)



    # Convert lists to numpy arrays for processing
    truth_pt    = np.concatenate(truth_pt, axis=0)
    truth_eta   = np.concatenate(truth_eta, axis=0)
    recon_pt    = np.concatenate(recon_pt, axis=0)
    recon_eta   = np.concatenate(recon_eta, axis=0)
    tru_pt      = np.asarray(tru_pt)
    tru_eta     = np.asarray(tru_eta)
    gnn_pt      = np.asarray(gnn_pt)
    gnn_eta     = np.asarray(gnn_eta)
    d_tru_pt    = np.asarray(d_tru_pt)
    d_tru_eta   = np.asarray(d_tru_eta)
    d_gnn_pt    = np.asarray(d_gnn_pt)
    d_gnn_eta   = np.asarray(d_gnn_eta)
    u_gnn_pt    = np.asarray(u_gnn_pt)
    u_gnn_eta   = np.asarray(u_gnn_eta)
    n_gnn_pt    = np.asarray(n_gnn_pt)
    n_gnn_eta   = np.asarray(n_gnn_eta)
    m_exa_pt    = np.asarray(m_exa_pt)
    m_exa_eta   = np.asarray(m_exa_eta)
    r_exa_pt    = np.asarray(r_exa_pt)
    r_exa_eta   = np.asarray(r_exa_eta)

    # Construct histograms
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    hist_truth_pt  = np.histogram(truth_pt,  bins=bins,  range=(.5,5))
    hist_tru_pt    = np.histogram(tru_pt,    bins=bins,  range=(.5,5))
    hist_gnn_pt    = np.histogram(gnn_pt,    bins=bins,  range=(.5,5))
    hist_truth_eta = np.histogram(truth_eta, bins=20, range=(-4,4))
    hist_tru_eta   = np.histogram(tru_eta,   bins=20, range=(-4,4))
    hist_gnn_eta   = np.histogram(gnn_eta,   bins=20, range=(-4,4))
    hist_d_tru_pt  = np.histogram(d_tru_pt,  bins=bins,  range=(.5,5))
    hist_d_gnn_pt  = np.histogram(d_gnn_pt,  bins=bins,  range=(.5,5))
    hist_u_gnn_pt  = np.histogram(u_gnn_pt,  bins=bins,  range=(.5,5))
    hist_n_gnn_pt  = np.histogram(n_gnn_pt,  bins=bins,  range=(.5,5))
    hist_d_tru_eta = np.histogram(d_tru_eta, bins=20, range=(-4,4))
    hist_d_gnn_eta = np.histogram(d_gnn_eta, bins=20, range=(-4,4))
    hist_u_gnn_eta = np.histogram(u_gnn_eta, bins=20, range=(-4,4))
    hist_n_gnn_eta = np.histogram(n_gnn_eta, bins=20, range=(-4,4))

    hist_recon_pt  = np.histogram(recon_pt,  bins=bins,  range=(.5,5))
    hist_recon_eta = np.histogram(recon_eta, bins=20, range=(-4,4))
    hist_m_exa_pt  = np.histogram(m_exa_pt,  bins=bins,  range=(.5,5))
    hist_m_exa_eta = np.histogram(m_exa_eta, bins=20, range=(-4,4))
    hist_r_exa_pt  = np.histogram(r_exa_pt,  bins=bins,  range=(.5,5))
    hist_r_exa_eta = np.histogram(r_exa_eta, bins=20, range=(-4,4))


    e_hist_truth_pt    = div(hist_truth_pt[0]  , np.sqrt(hist_truth_pt[0]))
    e_hist_tru_pt      = div(hist_tru_pt[0]    , np.sqrt(hist_tru_pt[0]))
    e_hist_gnn_pt      = div(hist_gnn_pt[0]    , np.sqrt(hist_gnn_pt[0]))
    e_hist_truth_eta   = div(hist_truth_eta[0] , np.sqrt(hist_truth_eta[0]))
    e_hist_tru_eta     = div(hist_tru_eta[0]   , np.sqrt(hist_tru_eta[0]))
    e_hist_gnn_eta     = div(hist_gnn_eta[0]   , np.sqrt(hist_gnn_eta[0]))
    e_hist_d_tru_pt    = div(hist_d_tru_pt[0]  , np.sqrt(hist_d_tru_pt[0]))
    e_hist_d_gnn_pt    = div(hist_d_gnn_pt[0]  , np.sqrt(hist_d_gnn_pt[0]))
    e_hist_u_gnn_pt    = div(hist_u_gnn_pt[0]  , np.sqrt(hist_u_gnn_pt[0]))
    e_hist_n_gnn_pt    = div(hist_n_gnn_pt[0]  , np.sqrt(hist_n_gnn_pt[0]))
    e_hist_d_tru_eta   = div(hist_d_tru_eta[0] , np.sqrt(hist_d_tru_eta[0]))
    e_hist_d_gnn_eta   = div(hist_d_gnn_eta[0] , np.sqrt(hist_d_gnn_eta[0]))
    e_hist_u_gnn_eta   = div(hist_u_gnn_eta[0] , np.sqrt(hist_u_gnn_eta[0]))
    e_hist_n_gnn_eta   = div(hist_n_gnn_eta[0] , np.sqrt(hist_n_gnn_eta[0]))

    e_hist_recon_pt    = div(hist_recon_pt[0]  , np.sqrt(hist_recon_pt[0]))
    e_hist_recon_eta   = div(hist_recon_eta[0] , np.sqrt(hist_recon_eta[0]))
    e_hist_m_exa_pt    = div(hist_m_exa_pt[0]  , np.sqrt(hist_m_exa_pt[0]))
    e_hist_m_exa_eta   = div(hist_m_exa_eta[0] , np.sqrt(hist_m_exa_eta[0]))
    e_hist_r_exa_pt    = div(hist_r_exa_pt[0]  , np.sqrt(hist_r_exa_pt[0]))
    e_hist_r_exa_eta   = div(hist_r_exa_eta[0] , np.sqrt(hist_r_exa_eta[0]))


    # Calculate efficiencies
    gra_pt_efficiency   = div(hist_tru_pt[0]    , hist_truth_pt[0])
    gnn_pt_efficiency   = div(hist_gnn_pt[0]    , hist_tru_pt[0])
    tot_pt_efficiency   = div(hist_gnn_pt[0]    , hist_truth_pt[0])
    gra_eta_efficiency  = div(hist_tru_eta[0]   , hist_truth_eta[0])
    gnn_eta_efficiency  = div(hist_gnn_eta[0]   , hist_tru_eta[0])
    tot_eta_efficiency  = div(hist_gnn_eta[0]   , hist_truth_eta[0])

    phys_pt_efficiency  = div(hist_m_exa_pt[0]  , hist_truth_pt[0])
    tech_pt_efficiency  = div(hist_r_exa_pt[0]  , hist_recon_pt[0])
    phys_eta_efficiency = div(hist_m_exa_eta[0] , hist_truth_eta[0])
    tech_eta_efficiency = div(hist_r_exa_eta[0] , hist_recon_eta[0])

    e_gra_pt_efficiency  = div_err(gra_pt_efficiency  ,e_hist_tru_pt  ,hist_tru_pt[0]  ,e_hist_truth_pt  ,hist_truth_pt[0])
    e_gnn_pt_efficiency  = div_err(gnn_pt_efficiency  ,e_hist_gnn_pt  ,hist_gnn_pt[0]  ,e_hist_tru_pt    ,hist_tru_pt[0])
    e_tot_pt_efficiency  = div_err(tot_pt_efficiency  ,e_hist_gnn_pt  ,hist_gnn_pt[0]  ,e_hist_truth_pt  ,hist_truth_pt[0])
    e_gra_eta_efficiency = div_err(gra_eta_efficiency ,e_hist_tru_eta ,hist_tru_eta[0] ,e_hist_truth_eta ,hist_truth_eta[0])
    e_gnn_eta_efficiency = div_err(gnn_eta_efficiency ,e_hist_gnn_eta ,hist_gnn_eta[0] ,e_hist_tru_eta   ,hist_tru_eta[0])
    e_tot_eta_efficiency = div_err(tot_eta_efficiency ,e_hist_gnn_eta ,hist_gnn_eta[0] ,e_hist_truth_eta ,hist_truth_eta[0])

    e_phys_pt_efficiency  = div_err(phys_pt_efficiency , e_hist_m_exa_pt  ,hist_m_exa_pt[0]  , e_hist_truth_pt , hist_truth_pt[0])
    e_tech_pt_efficiency  = div_err(tech_pt_efficiency , e_hist_r_exa_pt  ,hist_r_exa_pt[0]  , e_hist_recon_pt , hist_recon_pt[0])
    e_phys_eta_efficiency = div_err(phys_eta_efficiency, e_hist_m_exa_eta ,hist_m_exa_eta[0] , e_hist_truth_eta, hist_truth_eta[0])
    e_tech_eta_efficiency = div_err(tech_eta_efficiency, e_hist_r_exa_eta ,hist_r_exa_eta[0] , e_hist_recon_eta, hist_recon_eta[0])



    # Calculate fake fractions
    sum_tru_pt  = hist_tru_pt[0]  + hist_d_tru_pt[0]
    sum_gnn_pt  = hist_gnn_pt[0]  + hist_d_gnn_pt[0]  + hist_u_gnn_pt[0]  + hist_n_gnn_pt[0]
    sum_tru_eta = hist_tru_eta[0] + hist_d_tru_eta[0]
    sum_gnn_eta = hist_gnn_eta[0] + hist_d_gnn_eta[0] + hist_u_gnn_eta[0] + hist_n_gnn_eta[0]
    e_sum_tru_pt  = sum_err(e_hist_tru_pt,  e_hist_d_tru_pt)
    e_sum_gnn_pt  = sum_err(e_hist_gnn_pt,  e_hist_d_gnn_pt, e_hist_u_gnn_pt, e_hist_n_gnn_pt)
    e_sum_tru_eta = sum_err(e_hist_tru_eta, e_hist_d_tru_eta)
    e_sum_gnn_eta = sum_err(e_hist_gnn_eta, e_hist_d_gnn_eta, e_hist_u_gnn_eta, e_hist_n_gnn_eta)

    gra_pt_dfrac  = div(hist_d_tru_pt[0]  , sum_tru_pt)
    gnn_pt_dfrac  = div(hist_d_gnn_pt[0]  , sum_gnn_pt)
    gnn_pt_ufrac  = div(hist_u_gnn_pt[0]  , sum_gnn_pt)
    gnn_pt_nfrac  = div(hist_n_gnn_pt[0]  , sum_gnn_pt)
    gra_eta_dfrac = div(hist_d_tru_eta[0] , sum_tru_eta)
    gnn_eta_dfrac = div(hist_d_gnn_eta[0] , sum_gnn_eta)
    gnn_eta_ufrac = div(hist_u_gnn_eta[0] , sum_gnn_eta)
    gnn_eta_nfrac = div(hist_n_gnn_eta[0] , sum_gnn_eta)
    e_gra_pt_dfrac  = div_err(gra_pt_dfrac  ,e_hist_d_tru_pt  ,hist_d_tru_pt[0]  ,e_sum_tru_pt  ,sum_tru_pt)
    e_gnn_pt_dfrac  = div_err(gnn_pt_dfrac  ,e_hist_d_gnn_pt  ,hist_d_gnn_pt[0]  ,e_sum_gnn_pt  ,sum_gnn_pt)
    e_gnn_pt_ufrac  = div_err(gnn_pt_ufrac  ,e_hist_u_gnn_pt  ,hist_u_gnn_pt[0]  ,e_sum_gnn_pt  ,sum_gnn_pt)
    e_gnn_pt_nfrac  = div_err(gnn_pt_nfrac  ,e_hist_n_gnn_pt  ,hist_n_gnn_pt[0]  ,e_sum_gnn_pt  ,sum_gnn_pt)
    e_gra_eta_dfrac = div_err(gra_eta_dfrac ,e_hist_d_tru_eta ,hist_d_tru_eta[0] ,e_sum_tru_eta ,sum_tru_eta)
    e_gnn_eta_dfrac = div_err(gnn_eta_dfrac ,e_hist_d_gnn_eta ,hist_d_gnn_eta[0] ,e_sum_gnn_eta ,sum_gnn_eta)
    e_gnn_eta_ufrac = div_err(gnn_eta_ufrac ,e_hist_u_gnn_eta ,hist_u_gnn_eta[0] ,e_sum_gnn_eta ,sum_gnn_eta)
    e_gnn_eta_nfrac = div_err(gnn_eta_nfrac ,e_hist_n_gnn_eta ,hist_n_gnn_eta[0] ,e_sum_gnn_eta ,sum_gnn_eta)

    # Make the plots
    pt_bins  = [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.1, 1.3, 1.5, 1.7, 1.9, 2.25, 2.75, 3.25, 3.75, 4.5]
    pt_err   = [.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25, 0.25, 0.25, 0.5]
    eta_bins = hist_truth_eta[1] + .2
    eta_bins = eta_bins[eta_bins < 4]
    eta_err  = .2
    dpi=600
    file_root = group + '_' + model_group

    fig0, (ax0) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax0.hist(truth_pt, bins=bins, range=(0,5), histtype='step', edgecolor='blue',   fill=False, label='Truth Tracks')
    ax0.hist(tru_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='orange', fill=False, label='Graph Tracks')
    ax0.hist(gnn_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='green',  fill=False, label='Inferenced Tracks')
    ax0.legend(loc='upper right')
    ax0.set_xlabel('pT [GeV]')
    ax0.set_xlim(0, 5)
    ax0.set_ylabel('Particle Tracks')
    fig0.savefig(file_root + '_pt_counts.pdf', dpi=dpi)

    fig1, (ax1) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax1.hist(truth_eta, bins=20, range=(-4,4), histtype='step', edgecolor='blue',   fill=False, label='Truth Tracks')
    ax1.hist(tru_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='orange', fill=False, label='Graph Tracks')
    ax1.hist(gnn_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='green',  fill=False, label='Inferenced Tracks')
    ax1.legend(loc='upper right')
    ax1.set_xlabel('eta')
    ax1.set_xlim(-4, 4)
    ax1.set_ylabel('Particle Tracks')
    ax1.set_ylim(.9*hist_gnn_eta[0].min(), 1.2*hist_truth_eta[0].max())
    fig1.savefig(file_root + '_eta_counts.pdf', dpi=dpi)

    fig2, (ax2) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax2.errorbar(pt_bins, gra_pt_efficiency, xerr=pt_err, yerr=e_gra_pt_efficiency, fmt='o', label='Graph Construction Efficiency')
    ax2.errorbar(pt_bins, gnn_pt_efficiency, xerr=pt_err, yerr=e_gnn_pt_efficiency, fmt='o', label='GNN Inference Efficiency')
    ax2.errorbar(pt_bins, tot_pt_efficiency, xerr=pt_err, yerr=e_tot_pt_efficiency, fmt='o', label='Total Efficiency')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('pT [GeV]')
    ax2.set_xlim(0, 5)
    ax2.set_ylabel('Track Efficiency')
    ax2.set_ylim(0, 1)
    # ax2.set_ylim(.9, 1)
    fig2.savefig(file_root + '_pt_efficiencies.pdf', dpi=dpi)
    fig2.savefig(file_root + '_pt_efficiencies.png', dpi=dpi)

    fig3, (ax3) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax3.errorbar(eta_bins, gra_eta_efficiency, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='Graph Construction Efficiency')
    ax3.errorbar(eta_bins, gnn_eta_efficiency, xerr=eta_err, yerr=e_gnn_eta_efficiency, fmt='o', label='GNN Inference Efficiency')
    ax3.errorbar(eta_bins, tot_eta_efficiency, xerr=eta_err, yerr=e_tot_eta_efficiency, fmt='o', label='Total Efficiency')
    ax3.legend(loc='lower right')
    ax3.set_xlabel('eta')
    ax3.set_xlim(-4, 4)
    ax3.set_ylabel('Track Efficiency')
    # ax3.set_ylim(0, 1)
    ax3.set_ylim(.5, 1)
    # ax3.set_ylim(.8, 1)
    # ax3.set_ylim(.9, 1)
    fig3.savefig(file_root + '_eta_efficiencies.pdf', dpi=dpi)
    fig3.savefig(file_root + '_eta_efficiencies.png', dpi=dpi)
    with open(f'./{file_root}.pkl', 'wb') as fid:
        pickle.dump(fig3, fid)
    np.savez(file_root + '.npz', eta_bins, tot_eta_efficiency, eta_err, e_tot_eta_efficiency)


    fig4, (ax4) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax4.hist(d_tru_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='blue',   fill=False, label='Graph Construction: Duplicated Tracks')
    ax4.hist(d_gnn_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='orange', fill=False, label='GNN Inference: Duplicated Tracks')
    ax4.hist(u_gnn_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='green',  fill=False, label='GNN Inference: Mixed Tracks')
    # ax4.hist(n_gnn_pt,   bins=bins, range=(0,5), histtype='step', edgecolor='red',    fill=False, label='GNN Inference: Noise Tracks')
    ax4.legend(loc='upper right')
    ax4.set_xlabel('pT [GeV]')
    ax4.set_xlim(0, 5)
    ax4.set_ylabel('Fake Tracks')
    fig4.savefig(file_root + '_pt_fake_counts.pdf', dpi=dpi)

    fig5, (ax5) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax5.hist(d_tru_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='blue',   fill=False, label='Graph Construction: Duplicated Tracks')
    ax5.hist(d_gnn_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='orange', fill=False, label='GNN Inference: Duplicated Tracks')
    ax5.hist(u_gnn_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='green',  fill=False, label='GNN Inference: Mixed Tracks')
    # ax5.hist(n_gnn_eta,   bins=20, range=(-4,4), histtype='step', edgecolor='red',    fill=False, label='GNN Inference: Noise Tracks')
    ax5.legend(loc='upper center')
    ax5.set_xlabel('eta')
    ax5.set_xlim(-4, 4)
    ax5.set_ylabel('Fake Tracks')
    # ax5.set_ylim(.9*hist_gnn_eta[0].min(), 1.2*hist_truth_eta[0].max())
    fig5.savefig(file_root + '_eta_fake_counts.pdf', dpi=dpi)

    fig6, (ax6) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax6.errorbar(pt_bins, gra_pt_dfrac, xerr=pt_err, yerr=e_gra_pt_dfrac, fmt='o', label='Graph Construction: Duplication Fraction')
    ax6.errorbar(pt_bins, gnn_pt_dfrac, xerr=pt_err, yerr=e_gnn_pt_dfrac, fmt='o', label='GNN Inference: Duplication Fraction')
    ax6.errorbar(pt_bins, gnn_pt_ufrac, xerr=pt_err, yerr=e_gnn_pt_ufrac, fmt='o', label='GNN Inference: Mixed Track Fraction')
    # ax6.errorbar(pt_bins, gnn_pt_nfrac, xerr=pt_err, yerr=e_gnn_pt_nfrac, fmt='o', label='GNN Inference: Noise Track Fraction')
    ax6.legend(loc='best')
    ax6.set_xlabel('pT [GeV]')
    ax6.set_xlim(0, 5)
    ax6.set_ylabel('Fake Fraction')
    ax6.set_ylim(.001, .3)
    ax6.set_yscale('log')
    fig6.savefig(file_root + '_pt_fake_fractions.pdf', dpi=dpi)

    fig7, (ax7) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax7.errorbar(eta_bins, gra_eta_dfrac, xerr=eta_err, yerr=e_gra_eta_dfrac, fmt='o', label='Graph Construction: Duplication Fraction')
    ax7.errorbar(eta_bins, gnn_eta_dfrac, xerr=eta_err, yerr=e_gnn_eta_dfrac, fmt='o', label='GNN Inference: Duplication Fraction')
    ax7.errorbar(eta_bins, gnn_eta_ufrac, xerr=eta_err, yerr=e_gnn_eta_ufrac, fmt='o', label='GNN Inference: Mixed Track Fraction')
    # ax7.errorbar(eta_bins, gnn_eta_nfrac, xerr=eta_err, yerr=e_gnn_eta_nfrac, fmt='o', label='GNN Inference: Noise Track Fraction')
    ax7.legend(loc='best')
    ax7.set_xlabel('eta')
    ax7.set_xlim(-4, 4)
    ax7.set_ylabel('Fake Fraction')
    ax7.set_ylim(.001, .3)
    ax7.set_yscale('log')
    fig7.savefig(file_root + '_eta_fake_fractions.pdf', dpi=dpi)


    #ExaTrk plots
    fig8, (ax8) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax8.hist(truth_pt, bins=bins, range=(0,5), histtype='step', edgecolor='blue',   fill=False, label='Selected')
    ax8.hist(recon_pt, bins=bins, range=(0,5), histtype='step', edgecolor='orange', fill=False, label='Reconstructable')
    ax8.hist(m_exa_pt, bins=bins, range=(0,5), histtype='step', edgecolor='green',  fill=False, label='Matched')
    ax8.legend(loc='upper right')
    ax8.set_xlabel('pT [GeV]')
    ax8.set_xlim(0, 5)
    ax8.set_ylabel('Particles')
    fig8.savefig(file_root + '_pt_counts_exa.pdf', dpi=dpi)

    fig9, (ax9) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax9.hist(truth_eta, bins=20, range=(-4,4), histtype='step', edgecolor='blue',   fill=False, label='Selected')
    ax9.hist(recon_eta, bins=20, range=(-4,4), histtype='step', edgecolor='orange', fill=False, label='Reconstructable')
    ax9.hist(m_exa_eta, bins=20, range=(-4,4), histtype='step', edgecolor='green',  fill=False, label='Matched')
    ax9.legend(loc='upper right')
    ax9.set_xlabel('eta')
    ax9.set_xlim(-4, 4)
    ax9.set_ylabel('Particles')
    ax9.set_ylim(.9*hist_m_exa_eta[0].min(), 1.2*hist_truth_eta[0].max())
    fig9.savefig(file_root + '_eta_counts_exa.pdf', dpi=dpi)

    fig10, (ax10) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax10.errorbar(pt_bins, phys_pt_efficiency, xerr=pt_err, yerr=e_phys_pt_efficiency, fmt='o', label='Physics Efficiency')
    ax10.errorbar(pt_bins, tech_pt_efficiency, xerr=pt_err, yerr=e_tech_pt_efficiency, fmt='o', label='Technical Efficiency')
    ax10.legend(loc='lower right')
    ax10.set_xlabel('pT [GeV]')
    ax10.set_xlim(0, 5)
    ax10.set_ylabel('Track Efficiency')
    ax10.set_ylim(0, 1)
    # ax10.set_ylim(.75, 1)
    fig10.savefig(file_root + '_pt_efficiencies_exa.pdf', dpi=dpi)

    fig11, (ax11) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    ax11.errorbar(eta_bins, phys_eta_efficiency, xerr=eta_err, yerr=e_phys_eta_efficiency, fmt='o', label='Physics Efficiency')
    ax11.errorbar(eta_bins, tech_eta_efficiency, xerr=eta_err, yerr=e_tech_eta_efficiency, fmt='o', label='Technical Efficiency')
    ax11.legend(loc='lower right')
    ax11.set_xlabel('eta')
    ax11.set_xlim(-4, 4)
    ax11.set_ylabel('Track Efficiency')
    ax11.set_ylim(0, 1)
    # ax11.set_ylim(.75, 1)
    fig11.savefig(file_root + '_eta_efficiencies_exa.pdf', dpi=dpi)


    # fig12, (ax12) = plt.subplots(1, 1, dpi=dpi, figsize=(6, 6))
    # np.savez_compressed('/data/gnn_code/eta_phi.npz', gra_eta_efficiency)
    #
    # data = np.load('/data/gnn_code/eta_no_mm.npz')
    # gra_eta_efficiency_no_mm = data[data.files[0]]
    # data_1 = np.load('/data/gnn_code/eta_geo.npz')
    # gra_eta_efficiency_geo = data_1[data_1.files[0]]
    # data_3 = np.load('/data/gnn_code/eta_phi.npz')
    # gra_eta_efficiency_phi = data_3[data_3.files[0]]
    # # data_10 = np.load('/data/gnn_code/eta_mm_10.npz')
    # # gra_eta_efficiency_mm_10 = data_10[data_10.files[0]]
    # # data_30 = np.load('/data/gnn_code/eta_mm_30.npz')
    # # gra_eta_efficiency_mm_30 = data_30[data_30.files[0]]
    # # data_100 = np.load('/data/gnn_code/eta_mm_100.npz')
    # # gra_eta_efficiency_mm_100 = data_100[data_100.files[0]]
    # # data_300 = np.load('/data/gnn_code/eta_mm_300.npz')
    # # gra_eta_efficiency_mm_300 = data_300[data_300.files[0]]
    # # data_1000 = np.load('/data/gnn_code/eta_mm_1000.npz')
    # # gra_eta_efficiency_mm_1000 = data_1000[data_1000.files[0]]
    #
    # ax12.errorbar(eta_bins, gra_eta_efficiency_phi/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='phi on')
    # ax12.errorbar(eta_bins, gra_eta_efficiency_geo/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='phi and z0 on')
    #
    #
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_1/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 1')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_3/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 3')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_10/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 10')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_30/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 30')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_100/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 100')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_300/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 300')
    # # ax12.errorbar(eta_bins, gra_eta_efficiency_mm_1000/gra_eta_efficiency_no_mm, xerr=eta_err, yerr=e_gra_eta_efficiency, fmt='o', label='cut > 1000')
    #
    # ax12.legend(loc='lower right')
    # ax12.set_xlabel('eta')
    # ax12.set_xlim(-4, 4)
    # ax12.set_ylabel('Track Efficiency')
    # ax12.set_ylim(.96, 1)
    # fig12.savefig(file_root + '_eta_mm_efficiencies.pdf', dpi=dpi)
    # fig12.savefig(file_root + '_eta_mm_efficiencies.png', dpi=dpi)




    #Calculate and Print Totals
    N_truth     = np.sum(hist_truth_pt[0])
    N_graph     = np.sum(hist_tru_pt[0])
    N_gnn       = np.sum(hist_gnn_pt[0])
    N_d_graph   = np.sum(hist_d_tru_pt[0])
    N_d_gnn     = np.sum(hist_d_gnn_pt[0])
    N_unmatched = np.sum(hist_u_gnn_pt[0])
    N_noise     = np.sum(hist_n_gnn_pt[0])

    e_N_truth     = div(N_truth     , np.sqrt(N_truth))
    e_N_graph     = div(N_graph     , np.sqrt(N_graph))
    e_N_gnn       = div(N_gnn       , np.sqrt(N_gnn))
    e_N_d_graph   = div(N_d_graph   , np.sqrt(N_d_graph))
    e_N_d_gnn     = div(N_d_gnn     , np.sqrt(N_d_gnn))
    e_N_unmatched = div(N_unmatched , np.sqrt(N_unmatched))
    e_N_noise     = div(N_noise     , np.sqrt(N_noise))

    N_graph_clu = N_graph + N_d_graph
    N_gnn_clu   = N_gnn + N_d_gnn + N_unmatched + N_noise
    e_N_graph_clu = sum_err(e_N_graph, e_N_d_graph)
    e_N_gnn_clu   = sum_err(e_N_gnn,   e_N_d_gnn, e_N_unmatched, e_N_noise)

    gra_eff = div(N_graph , N_truth)
    gnn_eff = div(N_gnn   , N_graph)
    tot_eff = div(N_gnn   , N_truth)
    e_gra_eff = div_err(gra_eff ,e_N_graph, N_graph, e_N_truth, N_truth)
    e_gnn_eff = div_err(gnn_eff ,e_N_gnn  , N_gnn,   e_N_graph, N_graph)
    e_tot_eff = div_err(tot_eff ,e_N_gnn  , N_gnn,   e_N_truth, N_truth)

    gra_df = div(N_d_graph   , N_graph_clu)
    gnn_df = div(N_d_gnn     , N_gnn_clu)
    gnn_uf = div(N_unmatched , N_gnn_clu)
    gnn_nf = div(N_noise     , N_gnn_clu)
    e_gra_df = div_err(gra_df ,e_N_d_graph   ,N_d_graph,   e_N_graph_clu, N_graph_clu)
    e_gnn_df = div_err(gnn_df ,e_N_d_gnn     ,N_d_gnn,     e_N_gnn_clu,   N_gnn_clu)
    e_gnn_uf = div_err(gnn_uf ,e_N_unmatched ,N_unmatched, e_N_gnn_clu,   N_gnn_clu)
    e_gnn_nf = div_err(gnn_nf ,e_N_noise     ,N_noise,     e_N_gnn_clu,   N_gnn_clu)


    print('Track Efficiencies [-4,4]')
    print('Graph Construction Efficiency = ' +str(gra_eff) +' +/- ' +str(e_gra_eff))
    print('GNN Inference Efficiency      = ' +str(gnn_eff) +' +/- ' +str(e_gnn_eff))
    print('Total Efficiency              = ' +str(tot_eff) +' +/- ' +str(e_tot_eff))
    print()
    print('Fake Fractions [-4,4]')
    print('Graph Construction: Duplicate Track Fraction = ' +str(gra_df) +' +/- ' +str(e_gra_df))
    print('GNN Inference:      Duplicate Track Fraction = ' +str(gnn_df) +' +/- ' +str(e_gnn_df))
    print('GNN Inference:      Unmatched Track Fraction = ' +str(gnn_uf) +' +/- ' +str(e_gnn_uf))
    print('GNN Inference:      Noise     Track Fraction = ' +str(gnn_nf) +' +/- ' +str(e_gnn_nf))
    print('True Tracks: = ' + str(N_truth / n_events))
    print('Graph Clusters: = ' + str(N_graph_clu / n_events))
    print('GNN Track Candidates: = ' + str(N_gnn_clu / n_events))
    print()


    #Calculate and Print Totals (Barrel Region Only)
    N_truth     = np.sum(hist_truth_eta[0][5:15])
    N_graph     = np.sum(hist_tru_eta[0][5:15])
    N_gnn       = np.sum(hist_gnn_eta[0][5:15])
    N_d_graph   = np.sum(hist_d_tru_eta[0][5:15])
    N_d_gnn     = np.sum(hist_d_gnn_eta[0][5:15])
    N_unmatched = np.sum(hist_u_gnn_eta[0][5:15])
    N_noise     = np.sum(hist_n_gnn_eta[0][5:15])

    e_N_truth     = div(N_truth     , np.sqrt(N_truth))
    e_N_graph     = div(N_graph     , np.sqrt(N_graph))
    e_N_gnn       = div(N_gnn       , np.sqrt(N_gnn))
    e_N_d_graph   = div(N_d_graph   , np.sqrt(N_d_graph))
    e_N_d_gnn     = div(N_d_gnn     , np.sqrt(N_d_gnn))
    e_N_unmatched = div(N_unmatched , np.sqrt(N_unmatched))
    e_N_noise     = div(N_noise     , np.sqrt(N_noise))

    N_graph_clu = N_graph + N_d_graph
    N_gnn_clu   = N_gnn + N_d_gnn + N_unmatched + N_noise
    e_N_graph_clu = sum_err(e_N_graph, e_N_d_graph)
    e_N_gnn_clu   = sum_err(e_N_gnn,   e_N_d_gnn, e_N_unmatched, e_N_noise)

    gra_eff = div(N_graph , N_truth)
    gnn_eff = div(N_gnn   , N_graph)
    tot_eff = div(N_gnn   , N_truth)
    e_gra_eff = div_err(gra_eff ,e_N_graph, N_graph, e_N_truth, N_truth)
    e_gnn_eff = div_err(gnn_eff ,e_N_gnn  , N_gnn,   e_N_graph, N_graph)
    e_tot_eff = div_err(tot_eff ,e_N_gnn  , N_gnn,   e_N_truth, N_truth)

    gra_df = div(N_d_graph   , N_graph_clu)
    gnn_df = div(N_d_gnn     , N_gnn_clu)
    gnn_uf = div(N_unmatched , N_gnn_clu)
    gnn_nf = div(N_noise     , N_gnn_clu)
    e_gra_df = div_err(gra_df ,e_N_d_graph   ,N_d_graph,   e_N_graph_clu, N_graph_clu)
    e_gnn_df = div_err(gnn_df ,e_N_d_gnn     ,N_d_gnn,     e_N_gnn_clu,   N_gnn_clu)
    e_gnn_uf = div_err(gnn_uf ,e_N_unmatched ,N_unmatched, e_N_gnn_clu,   N_gnn_clu)
    e_gnn_nf = div_err(gnn_nf ,e_N_noise     ,N_noise,     e_N_gnn_clu,   N_gnn_clu)


    print('Track Efficiencies (Barrel Region Only [-2,2])')
    print('Graph Construction Efficiency = ' +str(gra_eff) +' +/- ' +str(e_gra_eff))
    print('GNN Inference Efficiency      = ' +str(gnn_eff) +' +/- ' +str(e_gnn_eff))
    print('Total Efficiency              = ' +str(tot_eff) +' +/- ' +str(e_tot_eff))
    print()
    print('Fake Fractions (Barrel Region Only [-2,2])')
    print('Graph Construction: Duplicate Track Fraction = ' +str(gra_df) +' +/- ' +str(e_gra_df))
    print('GNN Inference:      Duplicate Track Fraction = ' +str(gnn_df) +' +/- ' +str(e_gnn_df))
    print('GNN Inference:      Unmatched Track Fraction = ' +str(gnn_uf) +' +/- ' +str(e_gnn_uf))
    print('GNN Inference:      Noise     Track Fraction = ' +str(gnn_nf) +' +/- ' +str(e_gnn_nf))
    print()


    #Calculate and Print Totals (Endcaps Only)
    N_truth     = np.sum(hist_truth_eta[0][:5] + hist_truth_eta[0][15:])
    N_graph     = np.sum(hist_tru_eta  [0][:5] + hist_tru_eta  [0][15:])
    N_gnn       = np.sum(hist_gnn_eta  [0][:5] + hist_gnn_eta  [0][15:])
    N_d_graph   = np.sum(hist_d_tru_eta[0][:5] + hist_d_tru_eta[0][15:])
    N_d_gnn     = np.sum(hist_d_gnn_eta[0][:5] + hist_d_gnn_eta[0][15:])
    N_unmatched = np.sum(hist_u_gnn_eta[0][:5] + hist_u_gnn_eta[0][15:])
    N_noise     = np.sum(hist_n_gnn_eta[0][:5] + hist_n_gnn_eta[0][15:])

    e_N_truth     = div(N_truth     , np.sqrt(N_truth))
    e_N_graph     = div(N_graph     , np.sqrt(N_graph))
    e_N_gnn       = div(N_gnn       , np.sqrt(N_gnn))
    e_N_d_graph   = div(N_d_graph   , np.sqrt(N_d_graph))
    e_N_d_gnn     = div(N_d_gnn     , np.sqrt(N_d_gnn))
    e_N_unmatched = div(N_unmatched , np.sqrt(N_unmatched))
    e_N_noise     = div(N_noise     , np.sqrt(N_noise))

    N_graph_clu = N_graph + N_d_graph
    N_gnn_clu   = N_gnn + N_d_gnn + N_unmatched + N_noise
    e_N_graph_clu = sum_err(e_N_graph, e_N_d_graph)
    e_N_gnn_clu   = sum_err(e_N_gnn,   e_N_d_gnn, e_N_unmatched, e_N_noise)

    gra_eff = div(N_graph , N_truth)
    gnn_eff = div(N_gnn   , N_graph)
    tot_eff = div(N_gnn   , N_truth)
    e_gra_eff = div_err(gra_eff ,e_N_graph, N_graph, e_N_truth, N_truth)
    e_gnn_eff = div_err(gnn_eff ,e_N_gnn  , N_gnn,   e_N_graph, N_graph)
    e_tot_eff = div_err(tot_eff ,e_N_gnn  , N_gnn,   e_N_truth, N_truth)

    gra_df = div(N_d_graph   , N_graph_clu)
    gnn_df = div(N_d_gnn     , N_gnn_clu)
    gnn_uf = div(N_unmatched , N_gnn_clu)
    gnn_nf = div(N_noise     , N_gnn_clu)
    e_gra_df = div_err(gra_df ,e_N_d_graph   ,N_d_graph,   e_N_graph_clu, N_graph_clu)
    e_gnn_df = div_err(gnn_df ,e_N_d_gnn     ,N_d_gnn,     e_N_gnn_clu,   N_gnn_clu)
    e_gnn_uf = div_err(gnn_uf ,e_N_unmatched ,N_unmatched, e_N_gnn_clu,   N_gnn_clu)
    e_gnn_nf = div_err(gnn_nf ,e_N_noise     ,N_noise,     e_N_gnn_clu,   N_gnn_clu)


    print('Track Efficiencies (Endcaps Only)')
    print('Graph Construction Efficiency = ' +str(gra_eff) +' +/- ' +str(e_gra_eff))
    print('GNN Inference Efficiency      = ' +str(gnn_eff) +' +/- ' +str(e_gnn_eff))
    print('Total Efficiency              = ' +str(tot_eff) +' +/- ' +str(e_tot_eff))
    print()
    print('Fake Fractions (Endcaps Only)')
    print('Graph Construction: Duplicate Track Fraction = ' +str(gra_df) +' +/- ' +str(e_gra_df))
    print('GNN Inference:      Duplicate Track Fraction = ' +str(gnn_df) +' +/- ' +str(e_gnn_df))
    print('GNN Inference:      Unmatched Track Fraction = ' +str(gnn_uf) +' +/- ' +str(e_gnn_uf))
    print('GNN Inference:      Noise     Track Fraction = ' +str(gnn_nf) +' +/- ' +str(e_gnn_nf))
    print()



def get_particle_pt_eta(particle_id, track_attr):
    particle = track_attr[track_attr[:,0] == particle_id]
    # print(particle.shape)
    # print(particle[0,1], particle[0,2])
    return particle[0,1], particle[0,2]


def get_node_eta(pos):
    r = pos[0]
    z = pos[2]
    return np.arcsinh(z/r)


def div(a,b):
    return np.divide(a,b, out=np.zeros_like(a, dtype=float), where=b!=0)


def div_err(c, a_e, a, b_e, b):
    return c * np.sqrt(div(a_e,a)**2 + div(b_e,b)**2)


def sum_err(a, b, c=0, d=0):
    return np.sqrt(a**2 + b**2 + c**2 + d**2)

def find_all_paths(node_cluster, edge_index, edge_weights):
    # Create adjacency matrix for track candidate
    edges = []
    for j in range(len(node_cluster)):
        edges.append(edge_index[np.where(edge_index == node_cluster[j])[0]])
    edges = np.concatenate(edges, axis=0)
    if edges.shape[0] == 0: return []
    edges = np.unique(edges, axis=0)

    # Gather edge weights for track candidate
    weights = []
    for j in range(edges.shape[0]):
        weights.append( edge_weights[np.where(np.all(edge_index == edges[j], axis=1))] )
    weights = np.concatenate(weights, axis=0)

    # Check if any nodes have multiple neighbors, if not, algorithm finished
    ot_counts = np.asarray([np.count_nonzero(edges[:,0] == j) for j in node_cluster])
    in_counts = np.asarray([np.count_nonzero(edges[:,1] == j) for j in node_cluster])
    if (ot_counts[ot_counts > 1].shape[0] == 0 and in_counts[in_counts == 1].shape[0] > 0):
        # return [node_cluster], [list(-np.asarray(node_cluster))]
        return [node_cluster]

    # Find all starting and ending path nodes
    start = np.asarray(node_cluster)[in_counts == 0]
    end   = np.asarray(node_cluster)[ot_counts == 0]

    good_paths = []
    edge_scores = []
    for start_point in start:
        for end_point in end:
            all_paths = []
            path_weights = []
            path_queue   = [[start_point]]
            weight_queue = [[]]

            while len(path_queue) > 0:
                path   = path_queue.pop(0)
                weight = weight_queue.pop(0)

                next_step = edges[edges[:,0] == path[-1]][:,1]
                next_weight = weights[edges[:,0] == path[-1]]

                # remove any steps to nodes already in the path
                prune = []
                for i in range(len(next_step)):
                    if (next_step[i] in path):
                        prune.append(i)
                if len(prune) > 0:
                    prune = prune[::-1]
                    for i in prune:
                        next_step = np.delete(next_step, i)
                        next_weight = np.delete(next_weight, i)

                good_step = []
                if (len(next_weight) > 0):
                    # good_weight = next_weight[next_weight > 0.555]
                    good_weight = next_weight[next_weight > 0.8]
                    if (len(good_weight) > 0):
                        # good_step = next_step[next_weight > 0.555]
                        good_step = next_step[next_weight > 0.8]
                    else:
                        good_weight = [max(next_weight)]
                        # if (good_weight[0] > 0.555):
                        if (good_weight[0] > 0.3):
                            good_step = next_step[next_weight == good_weight[0]]

                if (len(good_step) > 0):
                    for i in range(len(good_weight)):
                        new_path = copy.deepcopy(path)
                        new_path.append(good_step[i])
                        new_weight = copy.deepcopy(weight)
                        new_weight.append(good_weight[i])
                        if good_step[i] == end_point:
                            all_paths.append(new_path)
                            path_weights.append(new_weight)
                        else:
                            path_queue.append(new_path)
                            weight_queue.append(new_weight)

            if len(all_paths) > 0:
                best_path = np.argmax(np.asarray([len(i) for i in all_paths]))             # Longest Path
                # best_path = np.argmax(np.asarray([sum(i) for i in path_weights]))          # Highest Total Score
                # best_path = np.argmax(np.asarray([sum(i) / len(i) for i in path_weights])) # Highest Average Score
                good_paths.append(all_paths[best_path])
                edge_scores.append(path_weights[best_path])

    # Use this chunk of code if you want to reduce to one track per cluster
    if len(good_paths) > 1:
        good_paths = [good_paths[np.argmax(np.asarray([len(i) for i in good_paths]))]]             # Longest Path
        # good_paths = [good_paths[np.argmax(np.asarray([sum(i) for i in edge_scores]))]]          # Highest Total Score
        # good_paths = [good_paths[np.argmax(np.asarray([sum(i) / len(i) for i in edge_scores]))]] # Highest Average Score

    # return good_paths, edge_scores
    return good_paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', '-g', default='test_no_edge', help='Group name of the inference graphs')
    parser.add_argument('--model', '-m', default='test_no_edge', help='Group name of the model to run on')
    parser.add_argument('--type', '-t', default='edge_classifier', help='Type of model to load')
    parser.add_argument('--reprocess', '-r', action='store_true', help='toggle reprocessing inference graphs')
    # parser.add_argument('--n_events', default=10, type=int, help='How many events to average over?')
    # parser.add_argument('--pt_min', default=1.5, type=float, help='lower pt range')
    # parser.add_argument('--pt_max', default=2.0, type=float, help='upper pt range')

    args = parser.parse_args()
    main(args)
