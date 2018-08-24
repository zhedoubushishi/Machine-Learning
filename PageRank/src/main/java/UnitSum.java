import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;


import java.io.IOException;
import java.text.DecimalFormat;

public class UnitSum {

    public static class PassMapper extends Mapper<Object, Text, Text, DoubleWritable> {

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            // input: toPage'\t'unitMultiplication
            // target: pass to reducer

            String[] line = value.toString().trim().split("\t");
            if (line.length < 2) {
                return;
            }

            context.write(new Text(line[0]), new DoubleWritable(Double.parseDouble(line[1])));
        }
    }

    public static class SumReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
            throws IOException, InterruptedException {

            // input key: toPage
            // input value: unitMultiplication

            // target: sum

            double sum = 0;

            for (DoubleWritable probability : values) {
                sum += probability.get();
            }

            DecimalFormat df = new DecimalFormat("#.00000");
            sum = Double.valueOf(df.format(sum));
            context.write(key, new DoubleWritable(sum));
        }
    }

    public static void main(String[] args) throws Exception{

        Configuration conf = new Configuration();

        Job job = Job.getInstance(conf);

        job.setJarByClass(UnitSum.class);

        job.setReducerClass(SumReducer.class);
        job.setMapperClass(PassMapper.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        job.waitForCompletion(true);
    }
}
