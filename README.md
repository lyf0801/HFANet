# HFANet
This is a repository about the paper "Hybrid Feature Aligned Network for Salient Object Detection in Optical Remote Sensing Imagery", accepted by IEEE TGRS 2022.

The code for HFANet will be available as soon as possible.

Note: The author is only a first-year postgraduate student, and due to research project reasons and copyright reasons, the code is in process of collation and the author will consider publishing it in due course, especially, the next papers based on this repository are under review or organized. 

Hope that the peers could understand and accommodate.

Note: About the Attribute Analysis in our paper, we use the following function to calculate SSIM metrics. 

![image](https://user-images.githubusercontent.com/73867361/186573197-aa017dd1-c599-43e9-8aa2-2e4c91eb5382.png)


        def _ssim(pred, gt):
                gt = gt.float()
                h, w = pred.size()[-2:]
                N = h * w
                x = pred.mean()
                y = gt.mean()
                sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
                sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
                sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

                aplha = 4 * x * y * sigma_xy
                beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

                if aplha != 0:
                    Q = aplha / (beta + 1e-20)
                elif aplha == 0 and beta == 0:
                    Q = 1.0
                else:
                    Q = 0
                return Q


Recently, we find that this function defines C1 = C2 = 0 to calculate SSIM metrics [while the original recommended C1=0.01 and C2 = 0.03](https://github.com/Shaosifan/HSENet/blob/main/codes/metric_scripts/calculate_PSNR_SSIM.py), and thus the reported variant of SSIM results is slightly different from the version for recommendation computation.

Hope that the peers could understand and accommodate.


If you have any questions about this paper, please send an e-mail to liuyanfeng99@gmail.com.

![image](https://user-images.githubusercontent.com/73867361/171971653-0bf8da14-1cd0-45e6-93c9-5980f910a5a1.png)
