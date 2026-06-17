package test.dr.evomodel.coalescent;

import dr.evolution.coalescent.DemographicFunction;
import dr.evolution.coalescent.IntervalType;
import dr.evolution.io.NewickImporter;
import dr.evolution.util.Units;
import dr.evomodel.bigfasttree.BigFastTreeIntervals;
import dr.evomodel.bigfasttree.BigFastTreeModel;
import dr.evomodel.coalescent.CoalescentLikelihood;
import dr.evomodel.coalescent.demographicmodel.ConstantPopulationModel;
import dr.evomodel.operators.UniformNodeHeightOperator;
import dr.evomodel.tree.TreeModel;
import dr.inference.model.Parameter;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Validates the incremental per-interval cache in CoalescentLikelihood against an
 * independent full recomputation of the original (cumulative-start-time) formula,
 * across a store -> propose -> getLogLikelihood -> accept/restore MCMC cycle that
 * mixes node-height moves and demographic (population-size) changes. The two must
 * agree to floating-point tolerance at every step; the small residual is the
 * absolute-vs-cumulative start-time rounding (mathematically identical).
 */
public class CoalescentLikelihoodIncrementalTest extends TestCase {

    private static String kingman(int n, long seed) {
        Random rng = new Random(seed);
        List<String> sub = new ArrayList<>(); List<Double> h = new ArrayList<>();
        for (int i = 0; i < n; i++) { sub.add("t" + i); h.add(0.0); }
        double t = 0.0; int k = n;
        while (k > 1) {
            t += -Math.log(rng.nextDouble()) / (k * (k - 1) / 2.0);
            int i = rng.nextInt(k); int j; do { j = rng.nextInt(k); } while (j == i);
            if (i > j) { int tmp = i; i = j; j = tmp; }
            sub.set(i, "(" + sub.get(i) + ":" + (t - h.get(i)) + "," + sub.get(j) + ":" + (t - h.get(j)) + ")");
            h.set(i, t); sub.remove(j); h.remove(j); k--;
        }
        return sub.get(0) + ";";
    }

    // Independent replica of the original cumulative-start-time full computation.
    private static double oracleFull(BigFastTreeIntervals intervals, ConstantPopulationModel demo) {
        int n = intervals.getIntervalCount();
        if (n == 0) return 0;
        double abs = intervals.getStartTime();
        demo.setTimeOffset(abs);
        DemographicFunction f = demo.getDemographicFunction();
        double startTime = 0, logL = 0;
        for (int i = 0; i < n; i++) {
            double dur = intervals.getInterval(i);
            double fin = startTime + dur;
            double area = f.getIntegral(startTime, fin);
            if (area == 0 && dur != 0) return Double.NEGATIVE_INFINITY;
            int lc = intervals.getLineageCount(i);
            double k2 = lc * (lc - 1) / 2.0;
            logL += -k2 * area;
            if (intervals.getIntervalType(i) == IntervalType.COALESCENT) {
                double d = f.getDemographic(fin);
                if (dur == 0.0 || d * (area / dur) >= f.getThreshold()) logL -= Math.log(d);
                else return Double.NEGATIVE_INFINITY;
            }
            startTime = fin;
        }
        return logL;
    }

    public void testIncrementalMatchesFull() throws Exception {
        CoalescentLikelihood.INCREMENTAL = true;
        int n = 250, iters = 30000;
        MathUtils.setSeed(5);
        TreeModel tree = new BigFastTreeModel(new NewickImporter(kingman(n, 5)).importTree(null));
        BigFastTreeIntervals iv = new BigFastTreeIntervals(tree);
        Parameter n0 = new Parameter.Default(5.0);
        ConstantPopulationModel demo = new ConstantPopulationModel(n0, Units.Type.YEARS);
        CoalescentLikelihood coal = new CoalescentLikelihood(iv, demo);
        UniformNodeHeightOperator nh = new UniformNodeHeightOperator(tree, 1.0);
        Random rng = new Random(71);

        boolean pass = true;
        int popMoves = 0;
        for (int it = 0; it < iters && pass; it++) {
            boolean accept = rng.nextBoolean();
            boolean popMove = rng.nextInt(8) == 0;
            double oldPop = n0.getParameterValue(0);
            tree.storeModelState(); iv.storeModelState(); coal.storeModelState();
            if (popMove) { n0.setParameterValue(0, 1.0 + rng.nextDouble() * 20.0); popMoves++; }
            else nh.doOperation();

            double lIncr = coal.getLogLikelihood();
            double lFull = oracleFull(iv, demo);
            double diff = Math.abs(lIncr - lFull);
            if (Double.isInfinite(lFull) && Double.isInfinite(lIncr)) diff = 0;
            if (diff > 1e-6 * (1 + Math.abs(lFull))) {
                System.out.printf("mismatch it=%d incr=%.10f full=%.10f popMove=%b%n", it, lIncr, lFull, popMove);
                pass = false;
            }
            if (accept) { tree.acceptModelState(); iv.acceptModelState(); coal.acceptModelState(); }
            else { tree.restoreModelState(); iv.restoreModelState(); coal.restoreModelState(); n0.setParameterValue(0, oldPop); }
        }
        assertTrue("demographic-change path not exercised", popMoves > 100);
        assertTrue(pass);
    }
}
