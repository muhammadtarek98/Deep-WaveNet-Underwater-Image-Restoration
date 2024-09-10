import cv2,torch,torchvision,torchinfo
from uie_euvp.models import CC_Module
from waternet.trail import transform_image,transform_array_to_image

if __name__ =="__main__":
    ckpt_dir="/home/muahmmad/projects/Image_enhancement/Deep-WaveNet-Underwater-Image-Restoration/uie_euvp/ckpts/netG_17.pt"
    state_dict=torch.load(f=ckpt_dir)
    #print(ckpt.keys())
    model=CC_Module()
    model.load_state_dict(state_dict=state_dict["model_state_dict"])
    model.eval()
    image = cv2.imread(
        filename="/home/muahmmad/projects/Image_enhancement/Enhancement_Dataset/9898_no_fish_f000130.jpg")
    image=cv2.cvtColor(src=image,code=cv2.COLOR_BGR2RGB)
    raw_input_tensor=transform_image(img=image)["X"]
    with torch.no_grad():
        pred=model(raw_input_tensor)
    pred=pred.squeeze_()
    pred=torch.permute(input=pred,dims=(1,2,0))
    pred=pred.detach().cpu().numpy()
    pred=transform_array_to_image(pred)
    cv2.imshow(winname="pred",mat=pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()