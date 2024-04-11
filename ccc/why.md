karpathy 用外行人觀點解釋為什麼要創造 llm.c 這一個專案

# 用外行術語解釋 llm.c

訓練大型語言模型 (LLM)（例如 ChatGPT）涉及大量程式碼和複雜性。

例如，典型的 LLM 訓練課程可能會使用 PyTorch 深度學習庫。 PyTorch 相當複雜，因為它實現了一個非常通用的張量抽象（一種排列和操作保存神經網路參數和激活的數字數組的方法），一個非常通用的反向傳播Autograd 引擎（訓練神經網路參數的演算法） ），以及您可能希望在神經網路中使用的大量深度學習層。 PyTorch 專案有 11,449 個檔案中的 3,327,184 行程式碼。

最重要的是，PyTorch 是用 Python 寫的，Python 本身就是一種非常高階的語言。 您必須執行 Python 解釋器將訓練程式碼轉換為低階電腦指令。 例如，執行此轉換的 cPython 專案包含 4,306 個檔案中的 2,437,955 行程式碼。

我正在刪除所有這些複雜性，並將 LLM 培訓簡化為其最基本的要素，以非常低階的語言 (C) 直接與電腦對話，並且沒有其他程式庫依賴項。 下面唯一的抽像是彙編程式碼本身。 我認為人們會感到驚訝的是，與上述相比，訓練像 GPT-2 這樣的 LLM 實際上只需要在單一檔案中使用大約 1000 行 C 程式碼。 我透過直接在 C 中實現 GPT-2 的神經網路訓練演算法來實現這種壓縮。這很困難，因為你必須詳細了解訓練演算法，能夠導出所有層的反向傳播的所有前向和後向傳遞，並非常仔細地實現所有數組索引計算，因為您沒有可用的PyTorch 張量抽象。 所以安排起來是一件非常脆弱的事情，但是一旦你這樣做了，並且通過再次檢查 PyTorch 來驗證正確性，你就會得到一些非常簡單、小而且在我看來相當漂亮的東西。

好吧，為什麼人們不一直這樣做呢？

第一：你放棄了很大的彈性。 如果你想改變你的神經網絡，在 PyTorch 中你可能需要改變一行程式碼。 在 llm.c 中，更改很可能會涉及更多程式碼，可能會更加困難，並且需要更多專業知識。 例如。 如果它是一個新的操作，你可能需要做一些微積分，並編寫它的前向傳播和後向傳播以進行反向傳播，並確保它在數學上是正確的。

第二：你正在放棄速度，至少一開始是這樣。 天下沒有免費的午餐 - 您不應該指望僅 1,000 行就能達到最先進的速度。 PyTorch 在後台做了很多工作，以確保神經網路非常有效率。 不僅所有 Tensor 操作都非常仔細地調用最高效的 CUDA 內核，而且還有 torch.compile 等功能，它可以進一步分析和優化您的神經網路以及它如何最有效地在您的電腦上運行。 現在，原則上，llm.c 應該能夠呼叫所有相同的核心並直接執行。 但這需要更多的工作和注意力，就像 (1) 一樣，如果您更改神經網路或正在運行的電腦的任何內容，您可能必須使用不同的參數調用不同的內核，並且您可能會手動進行更多更改。

簡要的說：llm.c是訓練GPT-2的直接實現。 這個實現結果出乎意料地短。 不支援其他神經網絡，僅支援 GPT-2，如果您想更改網絡的任何內容，則需要專業知識。 幸運的是，所有最先進的 LLM 實際上與 GPT-2 根本沒有太大的區別，因此這並不像您想像的那麼嚴格。 並且 llm.c 必須進行額外的調整和完善，但原則上我認為它應該能夠幾乎匹配（甚至超越，因為我們擺脫了所有開銷？）PyTorch，代碼不會比現代 LLM 多太多。

我為什麼要從事這份工作？ 因為這很有趣。 它也很有教育意義，因為只需要那 1,000 行非常簡單的 C 程式碼，沒有別的。 它只是一些數字數組和對其元素進行一些簡單的數學運算，例如 + 和 *。 對於正在進行的更多工作，它甚至可能變得實際有用。

英文原文

# explaining llm.c in layman terms
Training Large Language Models (LLMs), like ChatGPT, involves a large amount of code and complexity.

For example, a typical LLM training project might use the PyTorch deep learning library. PyTorch is quite complex because it implements a very general Tensor abstraction (a way to arrange and manipulate arrays of numbers that hold the parameters and activations of the neural network), a very general Autograd engine for backpropagation (the algorithm that trains the neural network parameters), and a large collection of deep learning layers you may wish to use in your neural network. The PyTorch project is 3,327,184 lines of code in 11,449 files.

On top of that, PyTorch is written in Python, which is itself a very high-level language. You have to run the Python interpreter to translate your training code into low-level computer instructions. For example the cPython project that does this translation is 2,437,955 lines of code across 4,306 files.

I am deleting all of this complexity and boiling the LLM training down to its bare essentials, speaking directly to the computer in a very low-level language (C), and with no other library dependencies. The only abstraction below this is the assembly code itself. I think people find it surprising that, by comparison to the above, training an LLM like GPT-2 is actually only a ~1000 lines of code in C in a single file. I am achieving this compression by implementing the neural network training algorithm for GPT-2 directly in C. This is difficult because you have to understand the training algorithm in detail, be able to derive all the forward and backward pass of backpropagation for all the layers, and implement all the array indexing calculations very carefully because you don’t have the PyTorch tensor abstraction available. So it’s a very brittle thing to arrange, but once you do, and you verify the correctness by checking agains PyTorch, you’re left with something very simple, small and imo quite beautiful.

Okay so why don’t people do this all the time?

Number 1: you are giving up a large amount of flexibility. If you want to change your neural network around, in PyTorch you’d be changing maybe one line of code. In llm.c, the change would most likely touch a lot more code, may be a lot more difficult, and require more expertise. E.g. if it’s a new operation, you may have to do some calculus, and write both its forward pass and backward pass for backpropagation, and make sure it is mathematically correct.

Number 2: you are giving up speed, at least initially. There is no fully free lunch - you shouldn’t expect state of the art speed in just 1,000 lines. PyTorch does a lot of work in the background to make sure that the neural network is very efficient. Not only do all the Tensor operations very carefully call the most efficient CUDA kernels, but also there is for example torch.compile, which further analyzes and optimizes your neural network and how it could run on your computer most efficiently. Now, in principle, llm.c should be able to call all the same kernels and do it directly. But this requires some more work and attention, and just like in (1), if you change anything about your neural network or the computer you’re running on, you may have to call different kernels, with different parameters, and you may have to make more changes manually.

So TLDR: llm.c is a direct implementation of training GPT-2. This implementation turns out to be surprisingly short. No other neural network is supported, only GPT-2, and if you want to change anything about the network, it requires expertise. Luckily, all state of the art LLMs are actually not a very large departure from GPT-2 at all, so this is not as strong of a constraint as you might think. And llm.c has to be additionally tuned and refined, but in principle I think it should be able to almost match (or even outperform, because we get rid of all the overhead?) PyTorch, with not too much more code than where it is today, for most modern LLMs.

And why I am working on it? Because it’s fun. It’s also educational, because those 1,000 lines of very simple C are all that is needed, nothing else. It's just a few arrays of numbers and some simple math operations over their elements like + and *. And it might even turn out to be practically useful with some more work that is ongoing.


