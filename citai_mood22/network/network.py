from citai_mood22.network.nn_blocks import *

class UNet( nn.Module ):
    def __init__(self, in_channels, out_channels=2, features=(32, 64, 128, 256, 320, 320), kernels=None,
                 conv_per_stage=2,
                 dropout_in_localization=False, out_in_stage=(True, True, True, True, False, False), block_kwargs={},
                 upsample=nn.ConvTranspose3d, outconv=nn.Conv3d, dropout_p=0.,):
        super( UNet, self ).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.num_pool = len( features )
        self.kernels = kernels
        self.conv_per_stage = conv_per_stage
        self.dropout_in_localization = dropout_in_localization
        self.out_in_stage = out_in_stage
        self.upsample = upsample
        self.outconv = outconv

        default_block_kwargs = {
            'conv': nn.Conv3d,
            'conv_kwargs': {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': False},
            'dropout_p': dropout_p,
            'norm': nn.BatchNorm3d,
            'norm_kwargs': {'eps': 1e-5, 'affine': True, "momentum": 0.1},
            'non_lin': nn.LeakyReLU,
            'non_lin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        }

        for k, v in block_kwargs.items():
            default_block_kwargs[k] = v

        # Feature extraction network
        self.d_blocks = []

        for i, f in enumerate( self.features ):
            for c in range( self.conv_per_stage ):

                kwargs = deepcopy( default_block_kwargs )

                if self.kernels is not None:
                    kwargs["conv_kwargs"]["kernel_size"] = kernels[i][f]
                    kwargs["conv_kwargs"]["padding"] = kernels[i][f] // 2

                # Input channels
                if i == 0 and c == 0:
                    self.d_blocks.append( ConvDropoutNormNonlin( self.in_channels, f, **kwargs ) )

                # Downsample in each stage
                elif i != 0 and c == 0:
                    kwargs["conv_kwargs"]["stride"] = 2
                    self.d_blocks.append( ConvDropoutNormNonlin( self.features[i - 1], f, **kwargs ) )

                # Any other down block
                else:
                    self.d_blocks.append( ConvDropoutNormNonlin( f, f, **kwargs ) )

        self.d_blocks = nn.ModuleList( self.d_blocks )

        # Localization network
        self.u_blocks = []

        rev_features = list( self.features )
        rev_features.reverse()

        for i, f in enumerate( rev_features[1:] ):  # Skips the 1st
            for c in range( self.conv_per_stage ):

                kwargs = deepcopy( default_block_kwargs )

                if self.kernels is not None:
                    kwargs["conv_kwargs"]["kernel_size"] = kernels[self.num_pool - i][self.conv_per_stage - f]
                    kwargs["conv_kwargs"]["padding"] = kernels[self.num_pool - i][self.conv_per_stage - f] // 2

                if not self.dropout_in_localization:
                    kwargs["dropout_p"] = 0

                # First in section: Add upsample (which makes features the same as skip) and Conv with 2x features
                if c == 0:
                    self.u_blocks.append( self.upsample( rev_features[i], f,
                                                         kernel_size = 2,
                                                         stride = 2,
                                                         bias = False ) )
                    self.u_blocks.append( ConvDropoutNormNonlin( f * 2, f, **kwargs ) )

                # Any other up block
                else:
                    self.u_blocks.append( ConvDropoutNormNonlin( f, f, **kwargs ) )

        # Create a Outputs at each resolution if required
        self.out_convs = {}
        for i, f in enumerate( rev_features[1:] ):
            if out_in_stage[:-1][-(i + 1)]:
                self.out_convs["out_" + str( i )] = self.outconv( f, self.out_channels, 1, 1, 0, 1, 1, bias = False )

        self.d_blocks = nn.ModuleList( self.d_blocks )
        self.u_blocks = nn.ModuleList( self.u_blocks )
        self.out_convs = nn.ModuleDict( self.out_convs )

        # Initialize weights by default, using the same approach as nnUnet
        self.apply( InitWeights_He() )


        ### Define loss function with deep-supervision
        weights = np.array( [1 / (2 ** i) for i in range( self.num_pool ) if self.out_in_stage[i]] )
        weights /= weights.sum()
        weights = np.flip( weights )
        self.loss_function = MultipleOutputLoss( torch.nn.BCEWithLogitsLoss(reduction='mean'), weights, interpolation = 'trilinear' )


    def forward_features(self, x):
        skips = []
        seg_outputs = []
        for i, d in enumerate( self.d_blocks ):

            # Keep the skips
            if (i != 0) and (i % self.conv_per_stage) == 0:
                skips.append( x )

            # Pass through down block
            x = d( x )

        for i, u in enumerate( self.u_blocks ):
            # Add the skips
            # print(u)
            if (i % (self.conv_per_stage + 1) == 1):
                x = torch.cat( [x, skips.pop()], dim = 1 )

            # Pass through up block
            x = u( x )

            # Append segmentation outputs at the end of each stage, if required
            out_n = i // (self.conv_per_stage + 1)
            if (i % (self.conv_per_stage + 1) == self.conv_per_stage) and \
                    self.out_in_stage[:-1][-(out_n + 1)]:
                seg_outputs.append( self.out_convs["out_" + str( out_n )]( x ) )

            # Intermediate outputs are provided at specific resolution, use the following to rescale
            # F.interpolate(x, size=label_size,mode='trilinear',align_corners=False)

        return seg_outputs

    def predict(self, x):
        """
        Only return the last reslution output
        """
        return self.forward_features(x)[-1]

    def forward(self, imgs, label):
        pred = self.forward_features( imgs )
        loss = self.loss_function( pred, label )
        return loss, pred[-1]


if __name__ == "__main__":
    model = UNet(1,1)
    with torch.no_grad():
        # loss, pred = model(torch.randn(2,1,160,160,160), torch.randn(2,1,160,160,160))
        pred = model.predict(torch.randn(3,1,160,160,160))