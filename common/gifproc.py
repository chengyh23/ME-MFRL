from PIL import Image, ImageSequence
import numpy as np

def make_white_transparent(image, alpha):
    # 将图片转换为numpy数组
    data = np.array(image)
    # 定义白色阈值，根据需要调整
    threshold = 250  # 255是纯白色，这个值可以根据你的图片调整
    data[:,:,3] = alpha
    white_pixels = (data[:,:,:3] == 255).all(axis=2)
    # 将白色像素的Alpha通道设置为0
    data[white_pixels, 3] = 0
    # 将numpy数组转换回图片
    transparent_image = Image.fromarray(data)
    return transparent_image

def gif_to_overlay_image_with_transparency(gif_path, output_path):
    # 打开GIF文件
    with Image.open(gif_path) as img:
        # 创建一个和第一帧相同尺寸的图片
        first_frame = img.copy().convert('RGBA')
        final_image = Image.new('RGBA', first_frame.size, (0, 0, 0, 0))
        
        # # 将第一帧添加到最终图片
        # final_image.paste(first_frame, (0, 0))

        # 遍历GIF的其余帧
        for idx, frame in enumerate(ImageSequence.Iterator(img)):
            # if idx > 10: continue
            # print(idx)
            if not idx in [37,38,39]: continue
            # if not idx in [5,20,38]: continue
            # 转换帧为RGBA模式
            frame = frame.convert('RGBA')
            
            # 将白色转换为透明
            frame = make_white_transparent(frame,(idx/40.0)**2 *(255-50) + 50)
            
            # 将当前帧叠加到最终图片上
            final_image = Image.alpha_composite(final_image, frame)
        
        # 保存合并后的图片
        final_image.save(output_path)

# 使用示例

gif_path = 'data/render/re-Rsrspure-no-eg-dqn_Xx400-3v1-0/4999/replay_29.gif'  # GIF文件路径
# output_path = 'common/fig/demo.png'  # 输出图片路径
output_path = 'common/fig/snapshot_38.png'  # 输出图片路径
gif_to_overlay_image_with_transparency(gif_path, output_path)


# gif_to_single_image(gif_path, output_path, max_frames)