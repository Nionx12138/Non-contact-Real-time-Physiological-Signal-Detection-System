import torch
from titan_complex_ssm import ComplexSSM
from titan_real_ssm import RealSSM

def convert_weights(complex_ckpt_path, real_ckpt_path):
    complex_model = ComplexSSM()
    ckpt = torch.load(complex_ckpt_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    complex_model.load_state_dict(state_dict)

    real_model = RealSSM()
    real_state = real_model.state_dict()

    real_state['fc_out.weight'] = complex_model.fc_out.weight.data
    real_state['fc_out.bias'] = complex_model.fc_out.bias.data

    c_block = complex_model.ssm_block
    real_state['ssm_block.proj_real.weight'] = c_block.proj_real.weight.data
    real_state['ssm_block.proj_real.bias'] = c_block.proj_real.bias.data
    real_state['ssm_block.proj_imag.weight'] = c_block.proj_imag.weight.data
    real_state['ssm_block.proj_imag.bias'] = c_block.proj_imag.bias.data

    real_state['ssm_block.A_real_param'] = c_block.A_real.data
    real_state['ssm_block.A_imag_param'] = c_block.A_imag.data

    B = c_block.B_complex.data
    real_state['ssm_block.B_real'] = B.real
    real_state['ssm_block.B_imag'] = B.imag

    C = c_block.C_complex.data
    real_state['ssm_block.C_real'] = C.real
    real_state['ssm_block.C_imag'] = C.imag

    dt = c_block.dt_predictor
    real_state['ssm_block.dt_predictor.mlp.0.weight'] = dt.mlp[0].weight.data
    real_state['ssm_block.dt_predictor.mlp.0.bias']   = dt.mlp[0].bias.data
    real_state['ssm_block.dt_predictor.mlp.2.weight'] = dt.mlp[2].weight.data
    real_state['ssm_block.dt_predictor.mlp.2.bias']   = dt.mlp[2].bias.data

    real_model.load_state_dict(real_state)
    torch.save({'model_state_dict': real_model.state_dict()}, real_ckpt_path)
    print(f'✅ 权重转换完成，实数模型已保存至 {real_ckpt_path}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python convert_weights.py <复数权重路径> <输出实数权重路径>")
        sys.exit(1)
    convert_weights(sys.argv[1], sys.argv[2])