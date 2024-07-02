import torch
x = torch.ones(103,2, 1)
# b = torch.tensor([1,1])
b = torch.ones(2,1)
d =x/b
print('x/b:',d.shape)
# 创建一个二维对角线矩阵
# matrix = torch.tensor([[1, 0, 0],
#                        [0, 2, 0],
#                        [0, 0, 3]])

# # 提取对角线元素
# diag_elements = torch.diag(matrix)
# print(diag_elements)
# # 将其转换为形状 (2, 1)
# # 注意，提取的对角线元素数量应与目标形状匹配
# # 这里对角线元素的数量是 3，因此我们将其转换为 (3, 1)
# result = diag_elements.view(-1, 1)

# print(result)
