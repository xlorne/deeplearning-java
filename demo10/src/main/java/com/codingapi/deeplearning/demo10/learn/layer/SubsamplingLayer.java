package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.core.InputType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class SubsamplingLayer implements NeuralNetworkLayer {

    private int[] kernelSizes;
    private int[] strides;

    //当前层数索引
    protected int index;
    //所有的网络层
    protected NeuralNetworkLayerBuilder layerBuilder;

    private int outSize;
    private InputType input;

    private INDArray a;


    private static  SubsamplingLayerBuilder builder = new SubsamplingLayerBuilder();

    public static SubsamplingLayerBuilder builder() {
        return builder;
    }

    protected SubsamplingLayer(int[] kernelSizes, int[] strides) {
        this.kernelSizes = kernelSizes;
        this.strides = strides;
    }


    @Override
    public INDArray forward(INDArray data) {

        int channel = input.getChannel();
        int patch = data.rows();

        INDArray a = Nd4j.create(patch,(channel*outSize*outSize));

        for(int i=0;i<patch;i++) {

            INDArray outDepthArray = Nd4j.create(channel,outSize*outSize);

            INDArray channelData = data.getRow(i).reshape(channel,input.getHeight()*input.getWidth());
            for(int c=0;c<channel;c++) {
                INDArray rowData = channelData.getRow(c).reshape(input.getHeight(),input.getWidth());
                INDArray outArray = Nd4j.create(outSize,outSize);
                for (int x = 0; x < outSize; x++) {
                    for (int y = 0; y < outSize; y++) {

                        INDArray item = rowData.get(NDArrayIndex.interval(x, x + kernelSizes[0]),
                                NDArrayIndex.interval(y, y + kernelSizes[1]));

                        Number sum = item.maxNumber();

                        outArray.put(x, y, sum);
                    }
                }
                outDepthArray.putRow(c,outArray.reshape(1,outSize*outSize));
            }

            a.putRow(i,outDepthArray.reshape(1,channel*outSize*outSize));
        }

        this.a = a;
        return a;
    }

    @Override
    public INDArray a() {
        return a;
    }

    @Override
    public LayerInitor initLayer(LayerInitor layerInitor) {

        input = layerInitor.getInputType();

        //{(n +2 x padding-filter) \over strides + 1}
        outSize = ((input.getHeight() + 2 - kernelSizes[0]) / strides[0] + 1 );

        return new LayerInitor(layerInitor.getLamdba(),layerInitor.getAlpha(),layerInitor.getSeed()
                ,new InputType(outSize,outSize,input.getChannel()));
    }

    @Override
    public void build(NeuralNetworkLayerBuilder layer, int index) {
        this.index = index;
        this.layerBuilder = layer;
    }

}
