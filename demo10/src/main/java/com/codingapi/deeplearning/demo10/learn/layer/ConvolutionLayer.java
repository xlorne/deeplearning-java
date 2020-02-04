package com.codingapi.deeplearning.demo10.learn.layer;

import com.codingapi.deeplearning.demo10.learn.activation.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * @author lorne
 * @date 2020/1/29
 * @description
 */
public class ConvolutionLayer extends BaseLayer {

    //管道数量
    private int channels;
    //内核大小
    private int[] kernelSizes;
    //卷积步长
    private int[] strides;
    //输出管道数量 对应filter的数量
    private int outChannels;
    //激活函数
    private Activation activation;
    //padding 大小
    private int[] padding;


    private List<INDArray> filters;

    private INDArray w;
    private INDArray b;

    private INDArray a;


    private INDArray convolution(INDArray data, INDArray filter) {
        //for(){}
        return null;
    }

    @Override
    public INDArray forward(INDArray data) {

        INDArray convolutionData =  Nd4j.empty();
        for(INDArray filter:filters){
            convolutionData.add(convolution(data,filter));
        }

        //z = w.Tx+b
        INDArray z = data.mmul(w).add(b.broadcast(data.rows(), b.columns()));

        a =  activation.activation(z);

        return a;
    }

    @Override
    public INDArray backprop(INDArray delta) {
        return null;
    }

    @Override
    public void init(double lamdba, double alpha, long seed) {
        filters = new ArrayList<>();
        for(int i = 0;i< outChannels;i++){
            filters.add(Nd4j.rand(kernelSizes,seed));
        }
        //
//        w = Nd4j.rand();
    }

    @Override
    public INDArray w() {
        return w;
    }

    @Override
    public INDArray a() {
        return a;
    }

    @Override
    public void updateParam() {

    }

    @Override
    public boolean isOutLayer() {
        return false;
    }

    protected ConvolutionLayer(int channels, int[] kernelSizes, int[] strides, int outChannels, Activation activation, int[] padding) {
        this.channels = channels;
        this.kernelSizes = kernelSizes;
        this.strides = strides;
        this.outChannels = outChannels;
        this.activation = activation;
        this.padding = padding;
    }

    private static ConvolutionLayerBuilder builder = new ConvolutionLayerBuilder();

    public static ConvolutionLayerBuilder builder(){
        return builder;
    }
}
