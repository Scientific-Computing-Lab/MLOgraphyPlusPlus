import torch.utils.data
import torch
import os
from models.model_single import ModelEmb
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.tbm import get_tbm_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from train import get_input_dict, norm_batch, get_dice_ji
import cv2
from cv_algorithms import guo_hall



def inference_ds(ds, model, sam, transform, epoch, args):
    pred_masks_folder = os.path.join(args['vis_folder'], "predicted_masks")
    gt_masks_folder = os.path.join(args['vis_folder'], "ground_truth_masks")
    os.makedirs(pred_masks_folder, exist_ok=True)
    os.makedirs(gt_masks_folder, exist_ok=True)
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    for ix, (imgs, gts, original_sz, img_sz, filename) in enumerate(pbar):
        filename = filename[0]
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts.unsqueeze(dim=0), input_size=input_size, original_size=original_size)
        masks = F.interpolate(masks, (Idim, Idim), mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0

        pred_mask_np = masks.squeeze().detach().cpu().numpy() * 255  
        gt_mask_np = gts.squeeze().detach().cpu().numpy() * 255  
        # Threshold and apply Guo-Hall thinning to the predicted mask
        _, pred_mask_np_thresh = cv2.threshold(pred_mask_np.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        guo_hall(pred_mask_np_thresh, inplace=True)

        # Threshold and apply Guo-Hall thinning to the ground truth mask
        _, gt_mask_np_thresh = cv2.threshold(gt_mask_np.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
        guo_hall(gt_mask_np_thresh, inplace=True)
        pred_mask_path_thinned = os.path.join(pred_masks_folder, "thinned_" + filename)
        gt_mask_path_thinned = os.path.join(gt_masks_folder, "thinned_" + filename)
        cv2.imwrite(pred_mask_path_thinned, pred_mask_np_thresh)
        cv2.imwrite(gt_mask_path_thinned, gt_mask_np_thresh)

        dice, ji = get_dice_ji(pred_mask_np_thresh / 255, gt_mask_np_thresh / 255)
        print(f"Image: {filename}, Post-thinning Dice: {dice:.4f}, Post-thinning IoU: {ji:.4f}")
        iou_list.append(ji)
        dice_list.append(dice)

        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)
            )
        )

    model.train()
    final_mean_dice = np.mean(dice_list)
    final_mean_iou = np.mean(iou_list)
    print(f"Final Mean Dice: {final_mean_dice:.4f}")
    print(f"Final Mean IoU: {final_mean_iou:.4f}")
    return final_mean_iou


def sam_call(batched_input, sam, dense_embeddings):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks



def main(args=None, sam_args=None):
    model = ModelEmb(args=args).cuda()
    model1 = torch.load(args['path_best'])
    model.load_state_dict(model1.state_dict())
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(sam_trans=transform)
    elif args['task'] == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
    elif args['task'] == 'tbm':
         trainset, testset = get_tbm_dataset(args, sam_trans=transform)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    with torch.no_grad():
        model.eval()
        inference_ds(ds_val, model.eval(), sam, transform, 0, args)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='tbm', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-folder', '--folder', default=6, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=256, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    parser.add_argument('--train_data_root', type=str, required=True, help='Path to the training data root directory')
    parser.add_argument('--test_data_root', type=str, required=True, help='Path to the testing data root directory')
    parser.add_argument('--sam_checkpoint', type=str, help='Path to SAM checkpoint')
    parser.add_argument('--model_type', type=str, default="vit_h", help='Model type for SAM (e.g., vit_h)')
    args = vars(parser.parse_args())
    args['path_best'] = os.path.join('results',
                                     'gpu' + str(args['folder']),
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + str(args['folder']), 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)

    sam_args = {
    'sam_checkpoint': args['sam_checkpoint'],
    'model_type': args['model_type'],
    'generator_args':{
        'points_per_side': 8,
        'pred_iou_thresh': 0.95,
        'stability_score_thresh': 0.7,
        'crop_n_layers': 0,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 0,
        'point_grids': None,
        'box_nms_thresh': 0.7,
    },
    'gpu_id': 0,
}

    main(args=args, sam_args=sam_args)