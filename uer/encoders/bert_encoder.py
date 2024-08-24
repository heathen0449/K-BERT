# -*- encoding:utf-8 -*-
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.layers.position_ffn import PositionwiseFeedForward
from uer.layers.multi_headed_attn import MultiHeadedAttention
from uer.layers.transformer import TransformerLayer


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args) for _ in range(self.layers_num)
        ])
        
    def forward(self, emb, seg, vm=None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            vm: [batch_size x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        seq_length = emb.size(1)
        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        """
        这段代码的作用是生成一个掩码（mask）用于后续的计算，通常用于在深度学习模型（如Transformer）中屏蔽某些不需要关注的部分。具体解释如下：

	1.	检查vm是否为None:
	•	如果vm为空，则根据seg生成掩码。
	•	如果vm不为空，则直接使用vm生成掩码。
	2.	当vm为空时:
	•	mask = (seg > 0)：根据seg生成一个布尔掩码，seg中大于0的位置为True，其余为False。
	•	unsqueeze(1)：在第二个维度（即索引为1的维度）增加一个维度。
	•	repeat(1, seq_length, 1)：将掩码在新增加的维度上重复seq_length次。
	•	unsqueeze(1)：再增加一个维度，使得掩码的形状为[batch_size x 1 x seq_length x seq_length]。
	•	mask = mask.float()：将布尔掩码转换为浮点型。
	•	mask = (1.0 - mask) * -10000.0：将True转换为0，将False转换为1，然后乘以-10000，使得不需要关注的部分值为-10000.0，表示在计算时需要屏蔽这些部分。
	3.	当vm不为空时:
	•	mask = vm.unsqueeze(1)：直接对vm增加一个维度，使得其形状为[batch_size x 1 x seq_length x seq_length]。
	•	mask = mask.float()：将掩码转换为浮点型。
	•	mask = (1.0 - mask) * -10000.0：同样将True转换为0，将False转换为1，然后乘以-10000，使得不需要关注的部分值为-10000.0。

这个掩码（mask）通常用于注意力机制中，确保模型只关注那些需要关注的部分，而屏蔽不需要关注的部分。
        """
        if vm is None:
            mask = (seg > 0). \
                    unsqueeze(1). \
                    repeat(1, seq_length, 1). \
                    unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0
        else:
            mask = vm.unsqueeze(1)
            mask = mask.float()
            mask = (1.0 - mask) * -10000.0

        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden
