from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage import gaussian_filter1d

# TO DO:
# 0. Zero detected people case for matching
# 2. Overlap
# 3. Batch input

def genPseudoColor(img_dp):
    avb = np.empty_like(img_dp)

    Q_neg_inf = (img_dp == 0)
    Q_pos_inf = (img_dp == (2 ** 16 - 1))
    Q_available = (~Q_neg_inf) & (~Q_pos_inf)

    avb[Q_neg_inf] = 0
    avb[Q_pos_inf] = 2 ** 8 - 1
    avb[Q_available] = int((2 ** 8 - 1) / 255)

    img_dp[Q_available] = np.clip(img_dp[Q_available] - 400, 0, 2 ** 12 - 1)
    img_dp[Q_neg_inf | Q_pos_inf] = 0

    img_dp = np.stack([img_dp // (2 ** 4), img_dp % (2 ** 8), avb], axis=-1)
    img_dp = np.clip(img_dp, 0, 255).astype(np.uint8)

    img_pseudo = cv2.rotate(img_dp, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img_pseudo

# Load a model
model = YOLO("./Training/runs/detect/train622/weights/best.pt")

thres_edge = 10     # the threshold for discarding targets
thres_vis = 40     # the threshold for discarding targets
thres_traverse = 200
thres_dep = 30
num_image = 600
conf = 0.7

delta_t = 1000/30  # unit: ms
H = np.diag([1, 1, 1, 0, 0, 0])
A = np.array([[1, 0, 0, delta_t, 0, 0],
              [0, 1, 0, 0, delta_t, 0],
              [0, 0, 1, 0, 0, delta_t],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
Q = np.diag([1000, 1000, 200, 1, 1, 1])

kalman_pre = []  # the current state of people
tar_list = []
tar_score = []   # the existence score of each target
tar_traj = []
traj_final = []
for i in range(1, num_image+1):
# for i in range(num_image, 0, -1):
    ppl_pts = []
    num_ppl_total = []  # the total number of detected people in each camera
    mea = []
    mea_var = [] # The measurements in this frame
    for j in range(2):
        # Read depth image
        # img_dep = cv2.imread(f'C:/Users/Haotian/Desktop/OneDrive_1_11-26-2023/5-two-overlap/sample/{i:06d}_{1 - j}_dp.png', -1)
        img_dep = cv2.imread(f'/Users/haotian/Downloads/3-new-new/sample/{i:06d}_{j}_dp.png', -1)
        # img_dep = cv2.imread(f'./Test/{i:06d}_{j}_dp.png', -1)
        img = genPseudoColor(np.copy(img_dep))
        img_dep = cv2.medianBlur(img_dep, 3)
        img_dep = img_dep.astype(float)
        img_dep[img_dep == 0] = np.nan
        img_dep[img_dep == 2 ** 16 - 1] = np.nan

        # YOLO Prediction
        results = model.predict(img, save=False, conf=conf, classes=0, max_det=3)

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # im.show()  # show image

        # Extract centers of people
        centers = []
        num_ppl = results[0].boxes.cls.size(0)
        num_ppl_total.append(num_ppl)
        for n in range(num_ppl):
            center_x = np.round(results[0].boxes.xywh[n, 0].item()).astype(int)
            center_y = np.round(results[0].boxes.xywh[n, 1].item()).astype(int)
            ppl_width = np.round(results[0].boxes.xywh[n, 2].item() / 16).astype(int)
            ppl_height = np.round(results[0].boxes.xywh[n, 3].item() / 20).astype(int)

            x_left = np.round(results[0].boxes.xyxy[n, 0].item()).astype(int)
            y_left = np.round(results[0].boxes.xyxy[n, 1].item()).astype(int)
            x_right = np.round(results[0].boxes.xyxy[n, 2].item()).astype(int)
            y_right = np.round(results[0].boxes.xyxy[n, 3].item()).astype(int)
            if x_left < thres_edge:
                center_x = 0
                center_y -= np.round(results[0].boxes.xywh[n, 3].item() / 6).astype(int)  # Lower the center
            if x_right > 480-thres_edge:
                center_x = 479
                center_y -= np.round(results[0].boxes.xywh[n, 3].item() / 6).astype(int) # Lower the center

            # Rotate the image back (clockwise 90 deg), (x, y) -> (h-y, x)
            x_min = np.max([0, center_x - ppl_width])
            x_max = np.min([center_x + ppl_width, 479])
            y_min = np.max([0, center_y - ppl_height])
            y_max = np.min([center_y + ppl_height, 639])

            x = np.linspace(x_min, x_max, x_max - x_min + 1)
            y = np.linspace(y_min, y_max, y_max - y_min + 1)
            xx, yy = np.meshgrid(640 - np.flip(y), x)

            x_min_rot = 640 - y_max
            x_max_rot = 640 - y_min
            y_min_rot = x_min
            y_max_rot = x_max

            # Extract the central region of detected people
            reg_dep = img_dep[y_min_rot:y_max_rot+1, x_min_rot:x_max_rot+1]
            bbox_dep = img_dep[640-y_right:640-y_left, x_left:x_right]
            center = [640 - center_y, center_x]

            # Delete outliers
            ref_dep_val = np.quantile(bbox_dep, 0.01)
            reg_dep[np.abs(reg_dep-ref_dep_val)>thres_dep] = np.nan
            if not np.all(np.isnan(reg_dep)):
                avg_dep = np.nanmean(reg_dep)  # Calculate the average depth
            else:
                num_ppl_total[-1] -= 1
                continue

            # De-distortion
            beta = 0.6
            if j == 0:
                cameraMatrix = np.load('./Calib/cameraMtx0.npy')
                distCoeffs = np.load('./Calib/distCoeffs0.npy')
            else:
                cameraMatrix = np.load('./Calib/cameraMtx1.npy')
                distCoeffs = np.load('./Calib/distCoeffs1.npy')

            img_pts = np.stack((xx.flatten(), yy.flatten()), axis=1)
            img_pts = np.squeeze(cv2.undistortImagePoints(img_pts.T, cameraMatrix, distCoeffs), axis=-2)
            img_pts = np.concatenate(
                [(img_pts - cameraMatrix[:2, -1]) / cameraMatrix.diagonal()[:2], np.ones((len(img_pts), 1))],
                axis=-1)
            img_pts /= np.linalg.norm(img_pts, axis=-1, keepdims=True)
            img_pts_dep = reg_dep.flatten()[..., None]
            img_pts = img_pts * img_pts_dep

            # Registration
            # Depth camera -> Microphone array
            img_pts = np.stack((img_pts[:, 1], img_pts[:, 2], img_pts[:, 0]), axis=1)
            if j == 0:
                theta = -np.deg2rad(90 + 35.6)
                Rota = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                img_pts[:, :2] = np.transpose(np.matmul(Rota, np.transpose(img_pts[:, :2])))
                mic_pts = img_pts + np.array([27.199, -57.347, -95])  # Translation
            else:
                theta = -np.deg2rad(90 - 35.6)
                Rota = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                img_pts[:, :2] = np.transpose(np.matmul(Rota, np.transpose(img_pts[:, :2])))
                mic_pts = img_pts + np.array([27.199, 57.347, -95])  # Translation

            ppl_pts.append(mic_pts)

            mean_pos = np.nanmean(mic_pts, axis=0)
            var_pos = np.nanvar(mic_pts, axis=0)
            mea.append(np.concatenate((mean_pos, np.array([0, 0, 0])), axis=0))
            mea_var.append(np.diag(np.concatenate((var_pos, np.array([1, 1, 0.1])), axis=0)))

    # plt.figure(figsize=(9, 12))
    # plt.ylim(-1500, 1500)
    # plt.xlim(0, 2000)
    # for n in range(np.sum(num_ppl_total)):
    #     plt.scatter(ppl_pts[n][:, 0], ppl_pts[n][:, 1], marker='+')


    # Predict
    for j in range(len(tar_list)):
        m, P = tar_list[j]
        m = A @ m
        P = A @ P @ A.T + Q
        tar_list[j] = (m, P)
        tar_traj[j].append([i, m, P])
        tar_score[j] -= 1

    # Matching
    # tar_list, mea, mea_var
    num_tar = len(tar_list)
    num_mea = len(mea)
    match_list = [-1 for k in range(num_mea)]
    same_mea = []
    if num_tar and num_mea:
        match_mtx = np.zeros((num_tar, num_mea))
        for jj in range(num_mea):
            for ii in range(num_tar):
                tar_pos = tar_list[ii][0][:]
                mea_pos = mea[jj][:]
                match_mtx[ii, jj] = np.linalg.norm(tar_pos-mea_pos)
            if np.abs(mea[jj][1]) < thres_traverse:
                same_mea.append(jj)
        min_val = np.min(match_mtx)
        while min_val != np.inf:
            row, col = np.where(match_mtx == min_val)
            row, col = row.item(), col.item()
            match_list[col] = row
            match_mtx[:, col] = np.inf
            match_mtx[row, :] = np.inf

            if col in same_mea:
                for pp in same_mea:
                    if pp != col and np.linalg.norm(mea[jj][:]-mea[pp][:])<350:
                        match_list[pp] = row
                        match_mtx[:, pp] = np.inf
            min_val = np.min(match_mtx)

    print(f'frame number: {i}')
    print(match_list)

    # Update
    for j in range(2):  # For each camera
        for n in range(num_ppl_total[j]):  # For each people
            idx = num_ppl_total[0]*j+n
            mea_ppl = mea[idx].T

            R = mea_var[idx]
            R[:3, :3] += np.exp(-(mea_ppl[1] / thres_traverse) ** 2 + 0.6) * (3000**2)*np.diag([10., 1., 1.])

            match_idx = match_list[idx]
            if match_idx >= 0:
                pre_ppl, P = tar_list[match_idx]

                r = mea_ppl - H @ pre_ppl
                S = np.linalg.inv(H @ P @ H.T + R)
                K = P @ H.T @ S

                pre_ppl = pre_ppl + K @ r
                P = P - P @ H.T @ S @ H @ P

                tar_list[match_idx] = (pre_ppl, P)
                if idx == num_mea-1 or (match_idx not in match_list[idx+1:]):
                    tar_score[match_idx] = np.min([tar_score[match_idx]+2, thres_vis])
                    tar_traj[match_idx][-1] = [i, pre_ppl, P]
            else:
                if np.abs(mea[idx][1])>thres_traverse:
                    tar_list.append((mea[idx].T, mea_var[idx]))
                    tar_score.append(thres_vis)
                    tar_traj.append([])
                    tar_traj[-1].append([i, mea[idx], mea_var[idx]])

    for pp, score in enumerate(tar_score):
        if score < 0:
            tar_list.pop(pp)
            traj_final.append(tar_traj[pp])
            tar_traj.pop(pp)
    tar_score = [score for score in tar_score if score >= 0]
    if i == num_image:
        for traj in tar_traj:
            traj_final.append(traj)

    # print(tar_score)
    # print(len(tar_list))

    # for pp in range(len(tar_list)):
    #     plt.scatter(tar_list[pp][0][0], tar_list[pp][0][1], marker='x')
    # plt.show()

    # rand_p = np.squeeze(np.linalg.cholesky(P) @ np.random.normal(size=(1000, 6, 1))) + kalman_pre[idx]
    # plt.scatter(rand_p[:, 0], rand_p[:, 1], marker='x')

    print('Yes')

# RTS smoother
for ii in range(len(traj_final)):
    for t in range(len(traj_final[ii]) - 2, 0, -1):
        i, m, P = traj_final[ii][t]
        _, m_, P_ = traj_final[ii][t + 1]

        G = np.linalg.inv(A @ P @ A.T + Q)
        m = m + P @ A.T @ G @ (m_ - A @ m)
        P = P + P @ A.T @ (G @ P_ @ G - G) @ A @ P

        traj_final[ii][t] = (i, m, P)

# Plot the trajectories
plt.figure(figsize=(9, 12))
plt.ylim(-2000, 2000)
plt.xlim(0, 1300)

traj_final_filtered = []
marker = itertools.cycle(('+', '.', 'o', '*', ','))
tra_idx = 0
for ii, traj in enumerate(traj_final):
    if len(traj) >= thres_vis * 2:
        x = np.array([traj[i][1][0] for i in range(len(traj))])
        y = np.array([traj[i][1][1] for i in range(len(traj))])
        t = np.array([traj[i][0] for i in range(len(traj))])
        plt.scatter(x, y, c=t, cmap='viridis', marker = next(marker))
        traj_final_filtered.append(traj)

        np.save(f'3_new_Trajectory_{tra_idx}', np.stack([t, x, y], axis=1))
        tra_idx += 1
plt.colorbar(label='t')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Trajectory')
plt.show()

print(len(traj_final_filtered))

# Save the filtered trajectories
# tra_array = np.squeeze(np.array(traj_final_filtered), axis=0)
# np.save('Trajectory_1', tra_array[:, :3])

print('over')