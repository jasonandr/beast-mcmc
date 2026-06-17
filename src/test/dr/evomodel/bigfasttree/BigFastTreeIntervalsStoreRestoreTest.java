package test.dr.evomodel.bigfasttree;

import dr.evolution.io.NewickImporter;
import dr.evomodel.bigfasttree.BigFastTreeIntervals;
import dr.evomodel.bigfasttree.BigFastTreeModel;
import dr.evomodel.operators.ScaleNodeHeightOperator;
import dr.evomodel.operators.UniformNodeHeightOperator;
import dr.evomodel.tree.TreeModel;
import dr.evomodelxml.operators.NodeHeightOperatorParser;
import dr.inference.operators.AdaptationMode;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Exercises the incremental store/restore path of BigFastTreeIntervals: drives
 * node-height proposals through a store -> propose -> recompute -> accept/restore
 * MCMC cycle and checks, after every step, that the intervals match a full
 * from-scratch rebuild. Covers both accepted and rejected (restored) states.
 */
public class BigFastTreeIntervalsStoreRestoreTest extends TestCase {

    private static String kingmanNewick(int n, long seed) {
        Random rng = new Random(seed);
        List<String> sub = new ArrayList<>();
        List<Double> h = new ArrayList<>();
        for (int i = 0; i < n; i++) { sub.add("t" + i); h.add(0.0); }
        double t = 0.0;
        int k = n;
        while (k > 1) {
            t += -Math.log(rng.nextDouble()) / (k * (k - 1) / 2.0);
            int i = rng.nextInt(k);
            int j; do { j = rng.nextInt(k); } while (j == i);
            if (i > j) { int tmp = i; i = j; j = tmp; }
            String merged = "(" + sub.get(i) + ":" + (t - h.get(i)) + "," + sub.get(j) + ":" + (t - h.get(j)) + ")";
            sub.remove(j); h.remove(j);
            sub.set(i, merged); h.set(i, t);
            k--;
        }
        return sub.get(0) + ";";
    }

    public void testStoreRestore() throws Exception {
        BigFastTreeIntervals.INCREMENTAL_STORE_RESTORE = true;
        int n = 250, iters = 20000;
        MathUtils.setSeed(42);
        TreeModel tree = new BigFastTreeModel(new NewickImporter(kingmanNewick(n, 42)).importTree(null));

        BigFastTreeIntervals iv = new BigFastTreeIntervals(tree);
        BigFastTreeIntervals ref = new BigFastTreeIntervals(tree);
        iv.calculateIntervals();

        UniformNodeHeightOperator nh = new UniformNodeHeightOperator(tree, 1.0);
        ScaleNodeHeightOperator root = new ScaleNodeHeightOperator(tree, 1.0, 0.75,
                NodeHeightOperatorParser.OperatorType.SCALEROOT, AdaptationMode.ADAPTATION_OFF, 0.25);
        Random rng = new Random(99);

        boolean pass = true;
        for (int it = 0; it < iters && pass; it++) {
            boolean accept = rng.nextBoolean();
            tree.storeModelState();
            iv.storeModelState();
            try {
                if (rng.nextInt(10) == 0) root.doOperation(); else nh.doOperation();
            } catch (Exception e) { /* internal reject */ }
            iv.getIntervalCount();
            if (accept) { tree.acceptModelState(); iv.acceptModelState(); }
            else { tree.restoreModelState(); iv.restoreModelState(); }

            // oracle: a fresh full rebuild from the current tree state
            ref.makeDirty();
            ref.calculateIntervals();
            if (iv.getIntervalCount() != ref.getIntervalCount()) { pass = false; break; }
            for (int i = 0; i < ref.getIntervalCount(); i++) {
                if (Math.abs(iv.getInterval(i) - ref.getInterval(i)) > 1e-9
                        || Math.abs(iv.getIntervalTime(i) - ref.getIntervalTime(i)) > 1e-9
                        || iv.getLineageCount(i) != ref.getLineageCount(i)) {
                    System.out.println("intervals store/restore mismatch at iter " + it + " interval " + i + " (accept=" + accept + ")");
                    pass = false;
                    break;
                }
            }
        }
        assertTrue(pass);
    }
}
