import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold.t_sne import _joint_probabilities
from sklearn import datasets

def guide_tsne_new(init, sim_metric, constraint, constraint_metric):
    instance_len = len(init)
    constraint_len = len(constraint)
    array_len = instance_len + len(constraint)
    distances = np.zeros(shape=(array_len, array_len))
    pos_param = np.zeros(shape=(array_len, array_len))
    for i in range(constraint_len, array_len):
        for j in range(constraint_len, array_len):
            distances[i][j] = 1 - sim_metric[i - constraint_len][j - constraint_len]
    for i in range(0, constraint_len):
        for j in range(constraint_len, array_len):
            if constraint_metric[i][j - constraint_len] > 0:
                pos_param[i][j] = pos_param[j][i] = constraint_metric[i][j - constraint_len]
            else:
                distances[i][j] = distances[j][i] = 1
    constraint.extend(init)
    X = np.array(constraint)
    M = np.array(distances)
    pos = np.array(pos_param).astype(np.float32, copy=False)
    tsne = TSNE(metric="precomputed", init=X, pos_param=pos)
    X_tsne = tsne.fit_transform(distances, skip_num_points=constraint_len)

    distances = 1 - sim_metric
    sum_pos = np.sum(constraint_metric)
    M = np.array(constraint[0:constraint_len])
    pos = np.array(constraint_metric / sum_pos).astype(np.float32, copy=False)
    tsne = TSNE(metric = "precomputed", init=X_tsne[constraint_len:,:], pos_param=pos, constraint=M)
    X_tsne = tsne.fit_transform(distances)

    return X_tsne

def incremental_tsne_new(init, sim_metric, constraint, constraint_metric, validated_list, changed_list):
    instance_len = len(init)
    maped_init = []
    map = {}
    pos1 = 0
    pos2 = instance_len - len(validated_list) - len(changed_list)
    stable_matrix = np.zeros(shape=(len(changed_list), instance_len))
    stable_index = 0
    for i in range(0, instance_len):
        if i in validated_list:
            map[pos2] = i
            pos2 += 1
        elif i in changed_list:
            map[pos2] = i
            constraint.extend(init[i])
            stable_matrix[stable_index][pos2] = 0.5
            pos2 += 1
        else:
            map[pos1] = i
            pos1 += 1
    for i in range(0, instance_len):
        maped_init.append(init[map[i]])

    constraint_metric.extend(stable_matrix)

    validated_list.extend(changed_list)
    array_len = instance_len + len(constraint)
    incremental_len = array_len - len(validated_list)
    distances = np.zeros(shape=(instance_len, instance_len))

    for i in range(0, instance_len):
        for j in range(0, instance_len):
            distances[i][j] = 1 - sim_metric[map[i]][map[j]]

    sum_pos = np.sum(constraint_metric)
    X = np.array(maped_init)
    pos = np.array(constraint_metric / sum_pos).astype(np.float32, copy=False)
    tsne = TSNE(metric="precomputed", init=X, pos_param=pos, constraint=constraint)
    X_tsne = tsne.fit_transform(distances, skip_num_points=incremental_len)

    return X_tsne

def guide_tsne_old(init, sim_metric, constraint, constraint_metric, alpha=1, cluster_num = 2, datatype="bird"):
    instance_len = len(init)
    constraint_len = len(constraint)
    array_len = instance_len+len(constraint)
    distances = np.zeros(shape=(array_len, array_len))
    pos_param = np.zeros(shape=(array_len, array_len))
    guidepos = {"dog": 0.8, "bird": 0.8, "bad_bird": 0.5}
    # dist_min = 1
    for i in range(constraint_len, array_len):
        for j in range(constraint_len, array_len):
            distances[i][j] = 1 - sim_metric[i-constraint_len][j-constraint_len]
            # if distances[i][j] < dist_min and i != j:
            #     dist_min = distances[i][j]
            # pos_param[i][j] = 0 #alpha
    for i in range(0, constraint_len):
        # for j in range(0, constraint_len):
        #     if i != j:
        #         distances[i][j] = distances[j][i] = 1
        for j in range(constraint_len, array_len):
            if constraint_metric[i][j-constraint_len] > 0:
                pos_param[i][j] = pos_param[j][i] = constraint_metric[i][j-constraint_len]
                # distances[i][j] = distances[j][i] = 0
            else:
                distances[i][j] = distances[j][i] = 1
    constraint.extend(init)
    X = np.array(constraint)
    M = np.array(distances)
    pos = np.array(pos_param).astype(np.float32, copy=False)
    tsne = TSNE(metric = "precomputed", init=X, pos_param=pos)
    X_tsne = tsne.fit_transform(distances, skip_num_points=constraint_len)
    for i in range(0, constraint_len):
        for j in range(constraint_len, array_len):
            if constraint_metric[i][j-constraint_len] > 0:
                distances[i][j] = distances[j][i] = guidepos[datatype] / cluster_num / pos_param[i][j]
    M = np.array(distances)
    tsne = TSNE(metric="precomputed", init=X_tsne)
    X_tsne = tsne.fit_transform(M, skip_num_points=constraint_len)
    # fig = plt.figure(figsize=(8, 8))
    # ax = plt.subplot(111)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    # for i in range(0, constraint_len):
    #      plt.text(X_tsne[i, 0], X_tsne[i, 1], str(i), family='serif', style='italic', ha='right', wrap=True)
    # plt.axis('tight')
    # plt.show()
    return X_tsne[constraint_len:,:]

def incremental_tsne_old(init, sim_metric, constraint, constraint_metric, validated_list, changed_list = [4,5], cluster_num = 2, datatype="bird"):
    incremental_list = validated_list
    instance_len = len(init)
    constraint_len = len(constraint)
    array_len = instance_len + len(constraint)
    incremental_len = array_len - len(incremental_list)
    distances = np.zeros(shape=(array_len, array_len))
    pos_param = np.zeros(shape=(array_len, array_len))
    guidepos = {"dog": 0.8, "bird": 0.8, "bad_bird": 0.5}
    maped_init = []
    map = {}
    pos1 = 0
    pos2 = instance_len - len(incremental_list)
    sum_pos = 0
    for i in range(0, instance_len):
        if i in incremental_list:
            map[pos2] = i
            pos2 += 1
        else:
            map[pos1] = i
            pos1 += 1
    for i in range(0, instance_len):
        maped_init.append(init[map[i]])
    for i in range(constraint_len, array_len):
        for j in range(constraint_len, array_len):
            distances[i][j] = 1 - sim_metric[map[i-constraint_len]][map[j-constraint_len]]
    for i in range(0, constraint_len):
        for j in range(constraint_len, array_len):
            if constraint_metric[i][map[j-constraint_len]] > 0:
                pos_param[i][j] = pos_param[j][i] = constraint_metric[i][map[j-constraint_len]]
                distances[i][j] = distances[j][i] = guidepos[datatype] / cluster_num / pos_param[i][j]
            else:
                distances[i][j] = distances[j][i] = 1
    constraint.extend(maped_init)
    X = np.array(constraint)
    M = np.array(distances)
    tsne = TSNE(metric="precomputed", init=X, method="exact")
    X_tsne = tsne.fit_transform(M, skip_num_points=incremental_len)


    for i in range(0, len(incremental_list)):
        init[incremental_list[i]] = X_tsne[incremental_len + i]

    return init

def guide_tsne(init, dis_metric, constraint, constraint_metric, cluster_num = 2, datatype="bird", multi_constraint=None):
    instance_len = len(init)
    constraint_len = len(constraint)
    array_len = instance_len+len(constraint)
    pos_param = np.zeros(shape=(array_len, array_len))
    distances = dis_metric.copy()

    for i in range(0, constraint_len):
        for j in range(constraint_len, array_len):
            if constraint_metric[i][j-constraint_len] > 0:
                pos_param[i][j] = pos_param[j][i] = constraint_metric[i][j-constraint_len]
                distances[i][j] = distances[j][i] = 0
            else:
                distances[i][j] = distances[j][i] = 1
    constraint.extend(init)
    X = np.array(constraint)
    pos = np.array(pos_param).astype(np.float32, copy=False)
    tsne = TSNE(metric = "precomputed", init=X, pos_param=pos)
    X_tsne = tsne.fit_transform(distances, skip_num_points=constraint_len)

    # max_sim = 0
    # for i in range(constraint_len, array_len):
    #     for j in range(constraint_len, array_len):
    #         if sim_metric[i][j] != 1 and sim_metric[i][j] >max_sim:
    #             max_sim = sim_metric[i][j]
    # multi_constraint = None
    if multi_constraint == None or cluster_num > 2:
        guidepos = {"dog": 2, "bird": 3, "bad_bird": 2.5}
        for i in range(0, constraint_len):
            for j in range(constraint_len, array_len):
                if constraint_metric[i][j-constraint_len] > 0:
                    distances[i][j] = distances[j][i] = min(dis_metric[i][j] * guidepos[datatype], 1)
        tsne = TSNE(metric="precomputed", init=X_tsne)
        X_tsne = tsne.fit_transform(distances, skip_num_points=constraint_len)
        return X_tsne[constraint_len:, :]
    else:
        guidepos = {"dog": 1.5, "bird": 3, "bad_bird": 2.2}
        for i in range(0, constraint_len):
            for j in range(constraint_len, array_len):
                if constraint_metric[i][j-constraint_len] > 0:
                    distances[i][j] = distances[j][i] = min(dis_metric[i][j] * guidepos[datatype], 1)
        P = _joint_probabilities(distances, 30, 0, False)
        pos_param = P[0:constraint_len,constraint_len:]
        pos_mean = pos_param.mean()
        for i in range(0, len(pos_param)):
            for j in range(0, len(pos_param[0])):
                if pos_param[i][j] < pos_mean:
                    pos_param[i][j] = 0
        tsne = TSNE(metric="precomputed", init=X_tsne[constraint_len:,:], constraint = np.array(multi_constraint), pos_param=pos_param)
        X_tsne = tsne.fit_transform(distances[constraint_len:,constraint_len:])
        return X_tsne

def incremental_tsne(init, dis_metric, constraint, constraint_metric, validated_list, changed_list, cluster_num = 2):
    instance_len = len(init)
    constraint_len = len(constraint)
    array_len = instance_len + constraint_len
    maped_init = []
    map = {}
    rmap = {}
    pos1 = 0
    pos2 = instance_len - len(validated_list) - len(changed_list)
    stable_matrix = np.zeros(shape=(instance_len))
    for i in range(0, instance_len):
        if i in validated_list:
            map[pos2] = i
            rmap[i] = pos2
            pos2 += 1
        elif i in changed_list:
            map[pos2] = i
            rmap[i] = pos2
            stable_matrix[pos2] = 0.5
            pos2 += 1
        else:
            map[pos1] = i
            pos1 += 1
    for i in range(0, instance_len):
        maped_init.append(init[map[i]])
    validated_list = np.concatenate((validated_list, changed_list),axis=0).astype(int)
    incremental_len = array_len - len(validated_list)

    distances = dis_metric.copy()

    for i in range(0, instance_len):
        for j in range(0, instance_len):
            distances[i + constraint_len][j + constraint_len] = dis_metric[map[i] + constraint_len][map[j] + constraint_len] #min(dis_metric[map[i] + constraint_len][map[j] + constraint_len] * 2, 1)

    constraint.extend(np.array(maped_init))
    tsne = TSNE(metric="precomputed", init=np.array(constraint), pos_param=stable_matrix, constraint = np.array(maped_init), n_iter=250) #method="exact",
    X_tsne = tsne.fit_transform(distances, skip_num_points=incremental_len)

    for i in range(0, len(validated_list)):
        init[validated_list[i]] = X_tsne[constraint_len + rmap[validated_list[i]]]

    return init

