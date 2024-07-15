from deep_differential_network.activations import *
from deep_differential_network.differential_hessian_network_ensemble import DifferentialNetwork


class QuadraticNetwork(DifferentialNetwork):
    name = "QuadraticNetwork"

    def __init__(self, n_input, **kwargs):
        # Compute In- / Output Sizes:
        self.m = int((n_input ** 2 + n_input) / 2)

        # Init the network:
        kwargs['n_output'] = self.m
        super(QuadraticNetwork, self).__init__(n_input, **kwargs)

        # Calculate the indices of the diagonal elements of L:
        self.diag_idx = np.arange(n_input) + 1
        self.diag_idx = self.diag_idx * (self.diag_idx + 1) / 2 - 1
        self.tri_idx = np.extract([x not in self.diag_idx for x in np.arange(self.m)], np.arange(self.m))
        self.tril_idx = np.tril_indices(self.n_input)

        self.s = nn.Softplus(beta=1)
        self.dsdz = SoftplusDer(beta=1.0)
        self.x0 = torch.zeros(1, n_input, 1)

        self.l = torch.zeros(self.n_network, 1, self.tril_idx[0].size, 1)
        self.L = torch.zeros(self.n_network, 1, self.n_input, self.n_input)
        self.dldz = torch.zeros(self.n_network, 1, self.n_input,  self.tril_idx[0].size)
        self.dLdz = torch.zeros(self.n_network, 1 * self.n_input, self.n_input, self.n_input)

    def forward(self, x):
        z = x.view(-1, self.n_input, 1)
        diff_z3 = z.view(1, -1, self.n_input, 1)
        diff_z4 = z.view(1, -1, 1, self.n_input, 1)

        # Construct L
        l_f, dldz_f = super(QuadraticNetwork, self).forward(z)
        dldz_f = dldz_f.transpose(dim0=2, dim1=3)

        l = self.l.repeat(1, x.shape[0], 1, 1)
        l[:, :, self.tri_idx] = l_f[:, :, self.tri_idx]
        l[:, :, self.diag_idx] = self.s(l_f[:, :, self.diag_idx]) + 1.e-3

        L = self.L.repeat(1, x.shape[0], 1, 1)
        L[:, :, self.tril_idx[0], self.tril_idx[1]] = l[:].view(self.n_network, -1, self.m)
        LT = L.transpose(dim0=2, dim1=3)

        # Construct H:
        H = torch.matmul(L, LT)

        # Construct the value function:
        H_diff_z3 = torch.matmul(H, diff_z3)
        V = -torch.matmul(diff_z3.transpose(dim0=2, dim1=3), H_diff_z3)

        # Construct dL/dz
        dldz = self.dldz.repeat(1, x.shape[0], 1, 1)
        dldz[..., self.tri_idx] = dldz_f[..., self.tri_idx]
        dldz[..., self.diag_idx] = self.dsdz(l_f[:, :, self.diag_idx]).transpose(dim0=2, dim1=3) * dldz_f[..., self.diag_idx]

        dLdz = self.dLdz.repeat(1, x.shape[0], 1, 1)
        dLdz[:, :, self.tril_idx[0], self.tril_idx[1]] = dldz.reshape(self.n_network, -1, self.m)
        dLdz = dLdz.view(self.n_network, -1, self.n_input, self.n_input, self.n_input)

        # Construct dH/dz
        dLdz_LT = torch.matmul(dLdz, LT.view(self.n_network, -1, 1, self.n_input, self.n_input))
        dHdz = dLdz_LT + dLdz_LT.transpose(3, 4)

        # Construct dV/dx
        dVdz = -2. * H_diff_z3 - torch.matmul(diff_z4.transpose(dim0=3, dim1=4), torch.matmul(dHdz, diff_z4)).view(self.n_network, -1, self.n_input, 1)
        dVdx = dVdz.transpose(dim0=2, dim1=3)

        return (V, dVdx)

    def cpu(self):
        super(QuadraticNetwork, self).cpu()
        self.x0 = self.x0.cpu()
        self.l = self.l.cpu()
        self.L = self.L.cpu()
        self.dldz = self.dldz.cpu()
        self.dLdz = self.dLdz.cpu()
        return self

    def cuda(self, device=None):
        super(QuadraticNetwork, self).cuda(device=device)
        self.x0 = self.x0.cuda()
        self.l = self.l.cuda()
        self.L = self.L.cuda()
        self.dldz = self.dldz.cuda()
        self.dLdz = self.dLdz.cuda()
        return self


class TrigonometricQuadraticNetwork(DifferentialNetwork):
    name = "TrigonometricQuadraticNetwork"

    def __init__(self, n_input, x_des = None, feature=None, **kwargs):

        self.x_des = x_des

        # set up the feature mask:
        feature = feature if feature is not None else torch.cat([torch.ones(1), torch.zeros(n_input-1)], dim=0)
        feature = np.clip(feature, 0., 1.0)
        # assert feature.size()[0] == n_input and torch.sum(feature) == 1.
        # self.idx = feature.argmax()
        self.indices = torch.nonzero(feature).squeeze()
        self.n_feature = n_input
        n_transform = int(feature.sum())
    
        # Compute In- / Output Sizes:
       
        self.m = int(((n_input + n_transform) ** 2 + (n_input + n_transform)) / 2)

        # Init the network:
        kwargs['n_output'] = self.m
        # super(TrigonometricQuadraticNetwork, self).__init__(self.n_feature + 1, **kwargs)
        super(TrigonometricQuadraticNetwork, self).__init__(self.n_feature + n_transform, **kwargs)

        # self.input is z,  self.n_feature is x, n_input is x 

        # Calculate the indices of the diagonal elements of L:
        self.diag_idx = np.arange(self.n_input) + 1
        self.diag_idx = self.diag_idx * (self.diag_idx + 1) / 2 - 1  #diag of x?
        self.tri_idx = np.extract([x not in self.diag_idx for x in np.arange(self.m)], np.arange(self.m))
        self.tril_idx = np.tril_indices(self.n_input)

        # Feature Mappings:
        # eye_f_input = torch.eye(self.n_feature - self.idx - 1)
        # eye_idx = torch.eye(self.idx)

        self.dzdx = torch.zeros(1, self.n_input, self.n_feature)

        # self.dzdx[:, :self.idx, :self.idx] = eye_idx
        # self.dzdx[:, self.idx + 2:, self.idx + 1:] = eye_f_input

        self.s = nn.Softplus(beta=1)
        self.dsdz = SoftplusDer(beta=1.0)

        # self.x0 = torch.zeros(1, n_input, 1)
        self.x0 = x_des.view(1, n_input, 1) if x_des is not None else torch.zeros(1, n_input, 1)
        print('x0', self.x0)
        
        self.z0, _ = self.z(self.x0)

        # self.l is the non-zero elements of L
        self.l = torch.zeros(self.n_network, 1, self.tril_idx[0].size, 1)
        self.L = torch.zeros(self.n_network, 1, self.n_input, self.n_input)
        self.dldz = torch.zeros(self.n_network, 1, self.n_input,  self.tril_idx[0].size)
        self.dLdz = torch.zeros(self.n_network, 1 * self.n_input, self.n_input, self.n_input)

    def forward(self, x):
        x = x.view(-1, self.n_feature, 1)

        z, dzdx = self.z(x)
        z = z.view(-1, self.n_input, 1)
        diff_z3 = (z - self.z0).view(1, -1, self.n_input, 1)
        diff_z4 = (z - self.z0).view(1, -1, 1, self.n_input, 1)

        # Construct L
        l_f, dldz_f = super(TrigonometricQuadraticNetwork, self).forward(z)
        dldz_f = dldz_f.transpose(dim0=2, dim1=3)

        l = self.l.repeat(1, x.shape[0], 1, 1)
        l[:, :, self.tri_idx] = l_f[:, :, self.tri_idx]
        l[:, :, self.diag_idx] = self.s(l_f[:, :, self.diag_idx]) + 1.e-3

        L = self.L.repeat(1, x.shape[0], 1, 1)
        L[:, :, self.tril_idx[0], self.tril_idx[1]] = l[:].view(self.n_network, -1, self.m)
        LT = L.transpose(dim0=2, dim1=3)

        # Construct H:
        H = torch.matmul(L, LT)

        # Construct the value function:
        H_diff_z3 = torch.matmul(H, diff_z3)
        V = -torch.matmul(diff_z3.transpose(dim0=2, dim1=3), H_diff_z3)

        # Construct dL/dz
        dldz = self.dldz.repeat(1, x.shape[0], 1, 1)
        dldz[..., self.tri_idx] = dldz_f[..., self.tri_idx]
        dldz[..., self.diag_idx] = self.dsdz(l_f[:, :, self.diag_idx]).transpose(dim0=2, dim1=3) * dldz_f[..., self.diag_idx]

        dLdz = self.dLdz.repeat(1, x.shape[0], 1, 1)
        dLdz[:, :, self.tril_idx[0], self.tril_idx[1]] = dldz.reshape(self.n_network, -1, self.m)
        dLdz = dLdz.view(self.n_network, -1, self.n_input, self.n_input, self.n_input)

        # Construct dH/dz
        dLdz_LT = torch.matmul(dLdz, LT.view(self.n_network, -1, 1, self.n_input, self.n_input))
        dHdz = dLdz_LT + dLdz_LT.transpose(3, 4)

        # Construct dV/dx
        dVdz = -2. * H_diff_z3 - torch.matmul(diff_z4.transpose(dim0=3, dim1=4),
                                              torch.matmul(dHdz, diff_z4)).view(self.n_network, -1, self.n_input, 1)

        dVdx = torch.matmul(dVdz.transpose(dim0=2, dim1=3), dzdx.unsqueeze(0))
        return (V, dVdx)

    def z(self, x):
        # Compute input transformation:
        # for i in self.indices:
        #     sin_th = torch.sin(x[:, i, 0])
        #     cos_th = torch.cos(x[:, i, 0])
    
        z = torch.zeros(x.shape[0], x.shape[1] + len(self.indices), x.shape[2],device=x.device)
        

        dzdx = self.dzdx.repeat(x.shape[0], 1, 1)

        z_row = 0
        for i in range(x.shape[1]):
            if i in self.indices:
                # 插入 sin(x[:,i]) 和 cos(x[:,i])
                z[:, z_row, 0] = torch.sin(x[:, i, 0])
                z[:, z_row + 1, 0] = torch.cos(x[:, i, 0])

                dzdx[:, z_row, i] = torch.cos(x[:, i, 0])  # sin 的导数是 cos
                dzdx[:, z_row + 1, i] = -torch.sin(x[:, i, 0])  # cos 的导数是 -sin

                z_row += 2
            else:
                
                z[:, z_row, 0] = x[:, i, 0]
                dzdx[:, z_row, i] = 1.0 

                z_row += 1

        # sin_th = torch.sin(x[:, self.idx, 0])
        # cos_th = torch.cos(x[:, self.idx, 0])

        # z = torch.cat((x[:, :self.idx], sin_th.view(-1, 1, 1), cos_th.view(-1, 1, 1), x[:, self.idx+1:]), dim=1)

        # dzdx = self.dzdx.repeat(x.shape[0], 1, 1)
        # dzdx[:, self.idx, self.idx] = cos_th
        # dzdx[:, self.idx + 1, self.idx] = -sin_th



        return z, dzdx

    def cpu(self):
        super(TrigonometricQuadraticNetwork, self).cpu()
        self.x0 = self.x0.cpu()
        self.z0 = self.z0.cpu()
        self.dzdx = self.dzdx.cpu()
        self.l = self.l.cpu()
        self.L = self.L.cpu()
        self.dldz = self.dldz.cpu()
        self.dLdz = self.dLdz.cpu()
        return self

    def cuda(self, device=None):
        if not torch.cuda.is_available():
            return self

        super(TrigonometricQuadraticNetwork, self).cuda(device=device)
        self.x0 = self.x0.cuda()
        self.z0 = self.z0.cuda()
        self.dzdx = self.dzdx.cuda()
        self.l = self.l.cuda()
        self.L = self.L.cuda()
        self.dldz = self.dldz.cuda()
        self.dLdz = self.dLdz.cuda()
        return self

# class TrigonometricQuadraticNetwork(DifferentialNetwork):
#     name = "TrigonometricQuadraticNetwork"

#     def __init__(self, n_input, feature=None, **kwargs):

#         # set up the feature mask:
#         feature = feature if feature is not None else torch.cat([torch.ones(1), torch.zeros(n_input-1)], dim=0)
#         feature = np.clip(feature, 0., 1.0)
#         # assert feature.size()[0] == n_input and torch.sum(feature) == 1.
#         self.idx = feature.argmax()
#         self.n_feature = n_input

#         # Compute In- / Output Sizes:
#         self.m = int(((n_input + 1) ** 2 + (n_input + 1)) / 2)

#         # Init the network:
#         kwargs['n_output'] = self.m
#         super(TrigonometricQuadraticNetwork, self).__init__(self.n_feature + 1, **kwargs)

#         # Calculate the indices of the diagonal elements of L:
#         print('n_input',n_input)
#         self.diag_idx = np.arange(n_input) + 1
#         self.diag_idx = self.diag_idx * (self.diag_idx + 1) / 2 - 1
#         self.tri_idx = np.extract([x not in self.diag_idx for x in np.arange(self.m)], np.arange(self.m))
#         self.tril_idx = np.tril_indices(self.n_input)

#         # Feature Mappings:
#         eye_f_input = torch.eye(self.n_feature - self.idx - 1)
#         eye_idx = torch.eye(self.idx)

#         self.dzdx = torch.zeros(1, self.n_input, self.n_feature)
#         self.dzdx[:, :self.idx, :self.idx] = eye_idx
#         self.dzdx[:, self.idx + 2:, self.idx + 1:] = eye_f_input

#         self.s = nn.Softplus(beta=1)
#         self.dsdz = SoftplusDer(beta=1.0)
#         self.x0 = torch.zeros(1, n_input, 1)
#         self.z0, _ = self.z(self.x0)

#         self.l = torch.zeros(self.n_network, 1, self.tril_idx[0].size, 1)
#         self.L = torch.zeros(self.n_network, 1, self.n_input, self.n_input)
#         self.dldz = torch.zeros(self.n_network, 1, self.n_input,  self.tril_idx[0].size)
#         self.dLdz = torch.zeros(self.n_network, 1 * self.n_input, self.n_input, self.n_input)

#     def forward(self, x):
#         x = x.view(-1, self.n_feature, 1)

#         z, dzdx = self.z(x)
#         z = z.view(-1, self.n_input, 1)
#         diff_z3 = (z - self.z0).view(1, -1, self.n_input, 1)
#         diff_z4 = (z - self.z0).view(1, -1, 1, self.n_input, 1)

#         # Construct L
#         l_f, dldz_f = super(TrigonometricQuadraticNetwork, self).forward(z)
#         dldz_f = dldz_f.transpose(dim0=2, dim1=3)

#         l = self.l.repeat(1, x.shape[0], 1, 1)
#         # print('self.tri_idx,',self.tri_idx)
#         # print('diag_idx',self.diag_idx)
#         l[:, :, self.tri_idx] = l_f[:, :, self.tri_idx]
#         l[:, :, self.diag_idx] = self.s(l_f[:, :, self.diag_idx]) + 1.e-3

#         L = self.L.repeat(1, x.shape[0], 1, 1)
#         L[:, :, self.tril_idx[0], self.tril_idx[1]] = l[:].view(self.n_network, -1, self.m)
#         LT = L.transpose(dim0=2, dim1=3)
#         # print('L',L.shape)

#         # Construct H:
#         H = torch.matmul(L, LT)

#         # Construct the value function:
#         H_diff_z3 = torch.matmul(H, diff_z3)
#         V = -torch.matmul(diff_z3.transpose(dim0=2, dim1=3), H_diff_z3)

#         # Construct dL/dz
#         dldz = self.dldz.repeat(1, x.shape[0], 1, 1)
#         dldz[..., self.tri_idx] = dldz_f[..., self.tri_idx]
#         dldz[..., self.diag_idx] = self.dsdz(l_f[:, :, self.diag_idx]).transpose(dim0=2, dim1=3) * dldz_f[..., self.diag_idx]

#         dLdz = self.dLdz.repeat(1, x.shape[0], 1, 1)
#         dLdz[:, :, self.tril_idx[0], self.tril_idx[1]] = dldz.reshape(self.n_network, -1, self.m)
#         dLdz = dLdz.view(self.n_network, -1, self.n_input, self.n_input, self.n_input)

#         # Construct dH/dz
#         dLdz_LT = torch.matmul(dLdz, LT.view(self.n_network, -1, 1, self.n_input, self.n_input))
#         dHdz = dLdz_LT + dLdz_LT.transpose(3, 4)

#         # Construct dV/dx
#         dVdz = -2. * H_diff_z3 - torch.matmul(diff_z4.transpose(dim0=3, dim1=4),
#                                               torch.matmul(dHdz, diff_z4)).view(self.n_network, -1, self.n_input, 1)

#         dVdx = torch.matmul(dVdz.transpose(dim0=2, dim1=3), dzdx.unsqueeze(0))
#         return (V, dVdx)

#     def z(self, x):
#         # Compute input transformation:
#         sin_th = torch.sin(x[:, self.idx, 0])
#         cos_th = torch.cos(x[:, self.idx, 0])

#         z = torch.cat((x[:, :self.idx], sin_th.view(-1, 1, 1), cos_th.view(-1, 1, 1), x[:, self.idx+1:]), dim=1)

#         dzdx = self.dzdx.repeat(x.shape[0], 1, 1)
#         dzdx[:, self.idx, self.idx] = cos_th
#         dzdx[:, self.idx + 1, self.idx] = -sin_th
#         return z, dzdx

#     def cpu(self):
#         super(TrigonometricQuadraticNetwork, self).cpu()
#         self.x0 = self.x0.cpu()
#         self.z0 = self.z0.cpu()
#         self.dzdx = self.dzdx.cpu()
#         self.l = self.l.cpu()
#         self.L = self.L.cpu()
#         self.dldz = self.dldz.cpu()
#         self.dLdz = self.dLdz.cpu()
#         return self

#     def cuda(self, device=None):
#         if not torch.cuda.is_available():
#             return self

#         super(TrigonometricQuadraticNetwork, self).cuda(device=device)
#         self.x0 = self.x0.cuda()
#         self.z0 = self.z0.cuda()
#         self.dzdx = self.dzdx.cuda()
#         self.l = self.l.cuda()
#         self.L = self.L.cuda()
#         self.dldz = self.dldz.cuda()
#         self.dLdz = self.dLdz.cuda()
#         return self