
public class Driver {

    public static void main(String[] args) throws Exception {

        //args[0]: dir of transition.txt
        //args[1]: dir of pageRank.txt
        //args[2]: dir of unitMultiplication output
        //args[3]: time of convergence
        //args[4]: beta

        UnitMultiplication multiplication = new UnitMultiplication();
        UnitSum sum = new UnitSum();

        String transitionMatrix = args[0];
        String prMatrix = args[1];
        String subPageRank = args[2];
        int count = Integer.parseInt(args[3]);

        for(int i = 0; i < count; i++) {
            String[] argsJob1 = {transitionMatrix, prMatrix + i, subPageRank + i};
            multiplication.main(argsJob1);

            String[] argsJob2 = {subPageRank + i, prMatrix + (i + 1)};
            sum.main(argsJob2);
         }
    }
}
