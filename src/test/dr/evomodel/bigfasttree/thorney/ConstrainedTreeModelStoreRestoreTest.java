package test.dr.evomodel.bigfasttree.thorney;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.bigfasttree.BigFastTreeModel;
import dr.evomodel.bigfasttree.thorney.ConstrainedTreeModel;
import dr.evomodel.bigfasttree.thorney.ConstrainedTreeOperator;
import dr.evomodel.operators.ExchangeOperator;
import dr.evomodel.operators.UniformNodeHeightOperator;
import dr.evomodel.tree.TreeModel;
import dr.inference.operators.AdaptationMode;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.Arrays;
import java.util.Random;

/**
 * Verifies the incremental store/restore in ConstrainedTreeModel (the tree model
 * actually sampled by the constrained Thorney BEAST pipeline). Uses a self-oracle:
 * the full tree state (every height, every parent edge, the root) is captured via
 * public getters before and after each proposal; after accept/restore the live tree
 * must equal the expected snapshot (post-proposal on accept, pre-proposal on
 * restore). Exercises both the height path and the topology (edges) path.
 */
public class ConstrainedTreeModelStoreRestoreTest extends TestCase {

    private static final String CONSTRAINTS = "((1:1,2:1,3:1,4:1,5:1):1,6:1);";
    private static final String BASE        = "(((((1:1,2:1):0.1,3:1):0.1,4:1):0.1,5:1):1,6:1);";

    private static int[] parents(TreeModel t) {
        int[] p = new int[t.getNodeCount()];
        for (int i = 0; i < t.getNodeCount(); i++) {
            NodeRef par = t.getParent(t.getNode(i));
            p[i] = par == null ? -1 : par.getNumber();
        }
        return p;
    }

    private static double[] heights(TreeModel t) {
        double[] h = new double[t.getNodeCount()];
        for (int i = 0; i < t.getNodeCount(); i++) h[i] = t.getNodeHeight(t.getNode(i));
        return h;
    }

    public void testStoreRestoreContract() throws Exception {
        ConstrainedTreeModel.INCREMENTAL_STORE_RESTORE = true;
        BigFastTreeModel.INCREMENTAL_STORE_RESTORE = true;
        MathUtils.setSeed(23);
        Tree constraints = new NewickImporter(CONSTRAINTS).importTree(null);
        TreeModel base = new BigFastTreeModel(new NewickImporter(BASE).importTree(null));
        ConstrainedTreeModel tree = new ConstrainedTreeModel("test", base, constraints);

        UniformNodeHeightOperator nh = new UniformNodeHeightOperator(tree, 1.0);
        ExchangeOperator narrow = new ExchangeOperator(0, null, 1.0);
        ConstrainedTreeOperator topo = new ConstrainedTreeOperator(tree, 1.0, narrow, 1.0, 1,
                AdaptationMode.ADAPTATION_OFF, 0.2);

        Random rng = new Random(23 * 11 + 5);
        int iters = 20000, topoChanged = 0;
        boolean pass = true;
        for (int it = 0; it < iters && pass; it++) {
            boolean accept = rng.nextBoolean();
            boolean doTopo = rng.nextInt(4) == 0;
            int[] pBefore = parents(tree); double[] hBefore = heights(tree); int rootBefore = tree.getRoot().getNumber();
            tree.storeModelState();
            try { if (doTopo) topo.doOperation(); else nh.doOperation(); } catch (Exception e) { /* internal reject */ }
            int[] pAfter = parents(tree); double[] hAfter = heights(tree); int rootAfter = tree.getRoot().getNumber();
            if (!Arrays.equals(pAfter, pBefore)) topoChanged++;

            int[] pExp; double[] hExp; int rootExp;
            if (accept) { tree.acceptModelState(); pExp = pAfter; hExp = hAfter; rootExp = rootAfter; }
            else { tree.restoreModelState(); pExp = pBefore; hExp = hBefore; rootExp = rootBefore; }

            if (!Arrays.equals(parents(tree), pExp) || !Arrays.equals(heights(tree), hExp)
                    || tree.getRoot().getNumber() != rootExp) {
                System.out.println("store/restore contract violated at iter " + it + " accept=" + accept + " doTopo=" + doTopo);
                pass = false;
            }
        }
        assertTrue("topology path not exercised", topoChanged > 100);
        assertTrue(pass);
    }
}
