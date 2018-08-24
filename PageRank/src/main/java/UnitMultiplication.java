import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.chain.ChainMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class UnitMultiplication {

    // calculate transition matrix mapper
    public static class TransitionMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //input format: fromPage'\t'toPage1,topPage2,toPage3...
            //target: create transition matrix unit -> fromPage'\t'toPage=probability
            //output key: formID
            //output value: toID = probability

            String[] line = value.toString().trim().split("\t");

            if (line.length == 1 || line[1].trim().equals("")) {
                return;
            }

            String fromPage = line[0];
            String[] toPages = line[1].split(",");

            for (String toPage : toPages) {
                StringBuilder sb = new StringBuilder();
                sb.append(toPage);
                sb.append('=');
                sb.append(String.valueOf(1.0 / toPages.length));
                context.write(new Text(fromPage), new Text(sb.toString()));
            }
        }
    }

    // PageRank Matrix Mapper
    public static class PRMapper extends Mapper<Object, Text, Text, Text> {

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            //input value: Page\tPageRank
            //target: write to

            //output key: PageID
            //output value: pageRank

            String[] line = value.toString().trim().split("\t");
            context.write(new Text(line[0]), new Text(line[1]));
        }
    }


    public static class MultiplicationReducer extends Reducer<Text, Text, Text, Text> {

        float beta;
        @Override
        public void setup(Context context) {
            Configuration conf = context.getConfiguration();
            beta = conf.getFloat("beta", 0.2f);
        }

        //input key: fromPage
        //input value: {toPage1=probability1, toPage2=probability2, ..., pageRank}

        //output key: toPage
        //output value: pageRank

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws
                IOException, InterruptedException {

            List<String> transitionUnit = new ArrayList<String>();
            double prUnit = 0;

            for (Text value : values) {
                if (value.toString().contains("=")) {
                    transitionUnit.add(value.toString());
                }
                else {
                    prUnit = Double.parseDouble(value.toString());
                }
            }

            for (String unit : transitionUnit) {
                String outputKey = unit.split("=")[0];
                double relation = Double.parseDouble(unit.split("=")[1]);

                String outputValue = String.valueOf(relation * prUnit * (1 - beta));  // + beta * ()

                context.write(new Text(outputKey), new Text(outputValue));
            }
        }
    }

    public static void main(String[] args) throws Exception {

        //arg[0]: transition.txt
        //arg[1]: pr.txt
        //arg[2]: output directory
        //arg[3]: beta

        Configuration conf = new Configuration();
        conf.setFloat("beta", Float.parseFloat(args[3]));

        Job job = Job.getInstance(conf);
        job.setJarByClass(UnitMultiplication.class);

        // chain two mapper
        ChainMapper.addMapper(job, TransitionMapper.class, Object.class, Text.class, Text.class, Text.class, conf);
        ChainMapper.addMapper(job, PRMapper.class, Object.class, Text.class, Text.class, Text.class, conf);

        job.setReducerClass(MultiplicationReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        MultipleInputs.addInputPath(job, new Path(args[0]), TextInputFormat.class, TransitionMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, PRMapper.class);

        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        job.waitForCompletion(true);

    }
}
