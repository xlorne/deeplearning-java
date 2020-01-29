package com.codingapi.deeplearning.demo10.learn.utils;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.esotericsoftware.kryo.serializers.JavaSerializer;
import org.objenesis.strategy.StdInstantiatorStrategy;
/**
 * 序列化工具类
 *
 * @author lorne
 *
 */
public class SerializeUtils {


    /**
     * 反序列化
     *
     * @param bytes 对象对应的字节数组
     * @return
     */
    public static <T> T deserialize(byte[] bytes,Class<T> clazz) {
        Input input = new Input(bytes);
        Kryo kryo = new Kryo();
        kryo.setInstantiatorStrategy(new Kryo.DefaultInstantiatorStrategy(new StdInstantiatorStrategy()));
        kryo.register(clazz, new JavaSerializer());
        Object obj = kryo.readClassAndObject(input);
        input.close();
        return (T)obj;
    }

    /**
     * 序列化
     *
     * @param object 需要序列化的对象
     * @return
     */
    public static byte[] serialize(Object object) {
        Output output = new Output(4096, -1);
        Kryo kryo = new Kryo();
        kryo.setInstantiatorStrategy(new Kryo.DefaultInstantiatorStrategy(new StdInstantiatorStrategy()));
        kryo.register(object.getClass(), new JavaSerializer());
        kryo.writeClassAndObject(output, object);
        byte[] bytes = output.toBytes();
        output.close();
        return bytes;
    }
}
