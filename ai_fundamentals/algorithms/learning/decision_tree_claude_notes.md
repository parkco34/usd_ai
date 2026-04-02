# BUILD TREE
Your tree actually built correctly. Let me walk you through the output by mapping it to the tree it created. The key is to read it **inside-out** — the deepest recursive calls print first, then unwind back up.

**Level 1 — Root split: Student background (IG = 0.537)**

The algorithm checked all 3 features and picked `Student background` as the best split. It has 3 unique values, so it loops through `CS`, `Maths`, `Other`:

**Branch: CS** → All 4 students with CS background passed. `y = [Pass, Pass, Pass, Pass]`. That's base case 1 (pure node). Returns a leaf immediately. Done.

**Branch: Other** → All 4 students with Other background failed. `y = [Fail, Fail, Fail, Fail]`. Pure node again. Leaf. Done.

**Branch: Maths** → Mixed: `y = [Pass, Fail, Fail, Pass, Pass, Pass, Fail]`. NOT pure. So it recurses deeper.

**Level 2 — Inside the Maths branch: Working Status (IG = 0.522)**

Now working with only the 7 Maths students. Best split is `Working Status`. Two values: `NW` and `W`:

**Branch: NW** → `y = [Pass, Pass, Pass]`. Pure. Leaf. Done.

**Branch: W** → `y = [Fail, Fail, Pass, Fail]`. Mixed. Recurses again.

**Level 3 — Inside Maths + Working: Other online courses (IG = 0.123)**

Down to 4 working Maths students. Only one feature left: `Other online courses`.

**Branch: N** → `y = [Fail, Pass, Fail]`. After deleting the column, `X` has 0 columns. Base case 2 fires. Majority is Fail. Leaf.

**Branch: Y** → `y = [Fail]`. Pure. Leaf.

Now the recursion **unwinds**. That's why you see the `Finally, the node =` prints appearing bottom-up — the deepest nodes finish first and return to their parents.

Your final tree reads like this:

```
Student background = CS?       → Pass
Student background = Other?    → Fail
Student background = Maths?
    Working Status = NW?       → Pass
    Working Status = W?
        Online courses = N?    → Fail
        Online courses = Y?    → Fail
```

The reason the output looks confusing is that your print statements show the work **in the order the computer does it** (depth-first), not in the order a human reads a tree (top-down). The `print_tree` function you wrote is specifically designed to fix that — it'll display it top-down with indentation. You have a small bug in it though: `value` should be `val` on the line that prints the branch label.
