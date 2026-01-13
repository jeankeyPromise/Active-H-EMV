from functools import cached_property
import time


print("=" * 60)
print("示例 1: 基础对比 - @property vs @cached_property")
print("=" * 60)

class Example:
    # 普通 @property：每次都计算
    @property
    def expensive_calc(self):
        print("  [expensive_calc] 正在计算...")
        time.sleep(0.5)  # 模拟耗时操作
        return sum(range(1000000))
    
    # @cached_property：只计算一次
    @cached_property
    def cached_calc(self):
        print("  [cached_calc] 正在计算...")
        time.sleep(0.5)  # 模拟耗时操作
        return sum(range(1000000))

obj = Example()

print("\n访问 expensive_calc (普通 @property):")
result1 = obj.expensive_calc
print(f"结果: {result1}")

print("\n再次访问 expensive_calc:")
result2 = obj.expensive_calc
print(f"结果: {result2}")

print("\n访问 cached_calc (@cached_property):")
result3 = obj.cached_calc
print(f"结果: {result3}")

print("\n再次访问 cached_calc:")
result4 = obj.cached_calc
print(f"结果: {result4} (注意：没有重新计算！)")


print("\n" + "=" * 60)
print("示例 2: 性能对比 - 多次访问的时间差异")
print("=" * 60)

class PerformanceExample:
    def __init__(self):
        self.property_call_count = 0
        self.cached_call_count = 0
    
    @property
    def slow_property(self):
        self.property_call_count += 1
        time.sleep(0.1)
        return sum(range(100000))
    
    @cached_property
    def fast_cached(self):
        self.cached_call_count += 1
        time.sleep(0.1)
        return sum(range(100000))

perf_obj = PerformanceExample()

# 测试普通 @property
print("\n测试 @property (访问10次):")
start = time.time()
for i in range(10):
    _ = perf_obj.slow_property
elapsed_property = time.time() - start
print(f"耗时: {elapsed_property:.3f}秒")
print(f"调用次数: {perf_obj.property_call_count}次")

# 测试 @cached_property
print("\n测试 @cached_property (访问10次):")
start = time.time()
for i in range(10):
    _ = perf_obj.fast_cached
elapsed_cached = time.time() - start
print(f"耗时: {elapsed_cached:.3f}秒")
print(f"调用次数: {perf_obj.cached_call_count}次")

print(f"\n性能提升: {elapsed_property/elapsed_cached:.1f}x 倍速！")


print("\n" + "=" * 60)
print("示例 3: 树形结构 - 模拟真实使用场景")
print("=" * 60)

class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
        self.leaf_access_count = 0
    
    @cached_property
    def all_leaves(self):
        """收集所有叶子节点的值（使用缓存）"""
        self.leaf_access_count += 1
        print(f"    正在收集节点 '{self.value}' 的叶子节点...")
        
        if not self.children:
            return [self.value]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves)
        return leaves

# 构建一个树形结构
#       Root
#      /  |  \
#     A   B   C
#    / \     / \
#   D   E   F   G

root = TreeNode("Root", [
    TreeNode("A", [
        TreeNode("D"),
        TreeNode("E")
    ]),
    TreeNode("B"),
    TreeNode("C", [
        TreeNode("F"),
        TreeNode("G")
    ])
])

print("\n第1次访问 all_leaves:")
leaves1 = root.all_leaves
print(f"结果: {leaves1}")

print("\n第2次访问 all_leaves:")
leaves2 = root.all_leaves
print(f"结果: {leaves2}")
print("(注意：没有重新计算！直接返回了缓存结果)")

print("\n第3次访问 all_leaves:")
leaves3 = root.all_leaves
print(f"结果: {leaves3}")


print("\n" + "=" * 60)
print("示例 4: 清除缓存")
print("=" * 60)

class CacheClearExample:
    def __init__(self):
        self.counter = 0
    
    @cached_property
    def dynamic_value(self):
        self.counter += 1
        print(f"  计算第 {self.counter} 次")
        return f"值-{self.counter}"

cache_obj = CacheClearExample()

print("\n第1次访问:")
print(f"结果: {cache_obj.dynamic_value}")

print("\n第2次访问:")
print(f"结果: {cache_obj.dynamic_value}")

print("\n清除缓存: del cache_obj.dynamic_value")
del cache_obj.dynamic_value

print("\n清除后再次访问:")
print(f"结果: {cache_obj.dynamic_value}")


print("\n" + "=" * 60)
print("示例 5: 不同实例有各自的缓存")
print("=" * 60)

class InstanceExample:
    def __init__(self, name):
        self.name = name
    
    @cached_property
    def expensive_result(self):
        print(f"  [{self.name}] 正在计算...")
        return f"{self.name} 的结果"

obj1 = InstanceExample("对象1")
obj2 = InstanceExample("对象2")

print("\n访问 obj1.expensive_result:")
print(f"结果: {obj1.expensive_result}")

print("\n访问 obj2.expensive_result:")
print(f"结果: {obj2.expensive_result}")

print("\n再次访问 obj1.expensive_result:")
print(f"结果: {obj1.expensive_result}")
print("(obj1 使用自己的缓存，没有重新计算)")

print("\n再次访问 obj2.expensive_result:")
print(f"结果: {obj2.expensive_result}")
print("(obj2 使用自己的缓存，没有重新计算)")


print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
@cached_property 的核心优势：
1. ✅ 自动缓存：第一次访问时计算，之后直接返回缓存
2. ✅ 语法简洁：像访问属性一样使用，无需手动管理缓存
3. ✅ 性能优化：避免重复计算，特别适合耗时操作
4. ✅ 实例隔离：每个实例有独立的缓存
5. ✅ 可清除：可以用 del 删除缓存重新计算

适用场景：
- 计算成本高的属性（如树遍历、复杂计算）
- 结果不会改变的属性（只读数据）
- 需要延迟计算的属性（用到时才计算）
""")