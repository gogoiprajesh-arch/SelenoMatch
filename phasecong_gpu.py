import torch
import torch.fft
import numpy as np
import math

def phasecong3_gpu(img_np, nscale=4, norient=6, minWaveLength=3, mult=1.6, sigmaOnf=0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = torch.tensor(img_np, dtype=torch.float32, device=device)
    rows, cols = img.shape
    
    y, x = torch.meshgrid(
        torch.linspace(-0.5, 0.5, rows, device=device),
        torch.linspace(-0.5, 0.5, cols, device=device),
        indexing='ij'
    )
    radius = torch.sqrt(x**2 + y**2)
    radius[radius == 0] = 1e-10  
    theta = torch.atan2(-y, x)
    
    radius = torch.fft.ifftshift(radius)
    theta = torch.fft.ifftshift(theta)
    
    imageFFT = torch.fft.fft2(img)
    
    EO = torch.zeros((norient, nscale, rows, cols), dtype=torch.complex64, device=device)
    pc_orient = torch.zeros((norient, rows, cols), dtype=torch.float32, device=device)
    
    for o in range(norient):
        angl = o * math.pi / norient
        
        ds = torch.sin(theta) * math.cos(angl) - torch.cos(theta) * math.sin(angl)
        dc = torch.cos(theta) * math.cos(angl) + torch.sin(theta) * math.sin(angl)
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread = torch.exp((-dtheta**2) / (2 * 1.2**2))
        
        sum_even = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        sum_odd = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        sum_an = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        
        for s in range(nscale):
            wavelength = minWaveLength * (mult ** s)
            fo = 1.0 / wavelength
            
            logGabor = torch.exp((-(torch.log(radius / fo))**2) / (2 * math.log(sigmaOnf)**2))
            logGabor[0, 0] = 0  
            
            filter_matrix = logGabor * spread
            
            EO[o, s] = torch.fft.ifft2(imageFFT * filter_matrix)
            
            even_resp = EO[o, s].real
            odd_resp = EO[o, s].imag
            amplitude = torch.sqrt(even_resp**2 + odd_resp**2)
            
            sum_even += even_resp
            sum_odd += odd_resp
            sum_an += amplitude
            
        energy = torch.sqrt(sum_even**2 + sum_odd**2)
        energy_clean = torch.clamp(energy - 1e-4, min=0.0)
        pc_orient[o] = energy_clean / (sum_an + 1e-7)

    covx2 = torch.zeros((rows, cols), device=device)
    covy2 = torch.zeros((rows, cols), device=device)
    covxy = torch.zeros((rows, cols), device=device)
    
    for o in range(norient):
        angl = o * math.pi / norient
        covx = pc_orient[o] * math.cos(angl)
        covy = pc_orient[o] * math.sin(angl)
        
        covx2 += covx**2
        covy2 += covy**2
        covxy += covx * covy
        
    covx2 = covx2 / (norient / 2)
    covy2 = covy2 / (norient / 2)
    covxy = 4 * covxy / norient
    
    denom = torch.sqrt((covxy**2 * 4) + (covx2 - covy2)**2)
    m = (covy2 + covx2 - denom) / 2
    
    return m.cpu().numpy(), EO.cpu().numpy()