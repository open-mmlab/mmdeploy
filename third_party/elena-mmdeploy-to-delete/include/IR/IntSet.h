#ifndef ELENA_INCLUDE_IR_INTSET_H_
#define ELENA_INCLUDE_IR_INTSET_H_

#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "IR/Expr.h"
#include "VisitorBase.h"

/**
 * TODO Zhichao, please Zhichao to add brief and details.
 */

class IntSet;
using IntSetPtr = std::shared_ptr<IntSet>;
using Rmap = std::unordered_map<ir::IterVarPtr, IntSetPtr>;

/// Define another format of ranges for IterVars
/// for computation convenience.
class IntSet {
 public:
  /// Default constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   IntSet();
  /// \encode
  ///
  /// \return Instance of class IntSet.
  IntSet();

  /// Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   IntSet(range);
  /// \encode
  ///
  /// \param range member of some IterVar instance;
  ///
  /// \return Instance of class IntSet.
  explicit IntSet(const ir::RangePtr& range);

  /// Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   IntSet(min, max);
  /// \encode
  ///
  /// \param min minimal boundary of the range
  ///  of some IterVar instance;
  /// \param min maximal boundary of the range
  ///  of some IterVar instance;
  ///
  /// \return Instance of class IntSet.
  IntSet(const ir::ExprPtr min, const ir::ExprPtr max);

  /// Set values of min and max with the input parameter.
  ///
  /// Typical Usage:
  /// \code
  ///   setRange(range_ptr);
  /// \encode
  ///
  /// \param r the range of some IterVar instance;
  ///
  /// \return None.
  void setRange(const ir::RangePtr& r);

  /// Set the values of both min and max
  /// equal to the input parameter.
  ///
  /// Typical Usage:
  /// \code
  ///   setSinglePoint(expr);
  /// \encode
  ///
  /// \param expr the expression used as the value;
  /// of min and max.
  ///
  /// \return None.
  void setSinglePoint(const ir::ExprPtr& expr);

  /// Set the value of both min and max
  /// to represent the range of positive infinity.
  ///
  /// Typical Usage:
  /// \code
  ///   setInf();
  /// \encode
  ///
  /// \return None.
  void setInf();

  /// Check if the range is empty.
  ///
  /// Typical Usage:
  /// \code
  ///   isEmpty();
  /// \encode
  ///
  /// \return true if the range is empty.
  bool isEmpty() const;

  /// Check if the range only contains a single point.
  ///
  /// Typical Usage:
  /// \code
  ///   isSinglePoint();
  /// \encode
  ///
  /// \return true if the range only contains a single point.
  bool isSinglePoint() const;

  /// Convert to the format of range.
  ///
  /// Typical Usage:
  /// \code
  ///   getRange();
  /// \encode
  ///
  /// \return instance of class Range.
  ir::RangePtr getRange() const;

  /// Get min_value.
  ///
  /// Typical Usage:
  /// \code
  ///   min();
  /// \encode
  ///
  /// \return min_value.
  ir::ExprPtr min() const;

  /// Get max_value.
  ///
  /// Typical Usage:
  /// \code
  ///   max();
  /// \encode
  ///
  /// \return max_value.
  ir::ExprPtr max() const;

  /// merge the range of two instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1.merge(s2);
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet merge(const IntSet& s) const;

  /// Operator + overloading, and perform
  /// operation between instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1 + s2;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator+(const IntSet& s) const;

  /// Operator - overloading, and perform
  /// operation between instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1 - s2;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator-(const IntSet& s) const;

  /// Operator * overloading, and perform
  /// operation between instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1 * s2;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator*(const IntSet& s) const;

  /// Operator % overloading, and perform
  /// operation between instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1 % s2;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator%(const IntSet& s) const;

  /// Operator / overloading, and perform
  /// operation between instances of class IntSet.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = IntSet();
  ///   s2 = IntSet();
  ///   s3 = s1 / s2;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator/(const IntSet& s) const;

  /// Operator + overloading, and perform
  /// operation between the instances of class IntSet
  /// and class Expr.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = api::constant<uint64_t>(8);
  ///   s2 = IntSet();
  ///   s3 = s2 + s1;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator+(const ir::ExprPtr& expr) const;

  /// Operator - overloading, and perform
  /// operation between the instances of class IntSet
  /// and class Expr.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = api::constant<uint64_t>(8);
  ///   s2 = IntSet();
  ///   s3 = s2 - s1;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator-(const ir::ExprPtr& expr) const;

  /// Operator * overloading, and perform
  /// operation between the instances of class IntSet
  /// and class Expr.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = api::constant<uint64_t>(8);
  ///   s2 = IntSet();
  ///   s3 = s2 * s1;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator*(const ir::ExprPtr& expr) const;

  /// Operator / overloading, and perform
  /// operation between the instances of class IntSet
  /// and class Expr.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = api::constant<uint64_t>(8);
  ///   s2 = IntSet();
  ///   s3 = s2 / s1;
  /// \encode
  ///
  /// \return instance of IntSet.
  IntSet operator/(const ir::ExprPtr& expr) const;

  /// Operator % overloading, and perform
  /// operation between the instances of class IntSet
  /// and class Expr.
  ///
  /// Typical Usage:
  /// \code
  ///   s1 = api::constant<uint64_t>(8);
  ///   s2 = IntSet();
  ///   s3 = s2 % s1;
  /// \encode
  ///
  /// \param expr the second operand.
  ///
  /// \return instance of IntSet.
  IntSet operator%(const ir::ExprPtr& expr) const;

  /// Use the newest values of IterVars to update
  /// the values of min and max.
  ///
  /// Typical Usage:
  /// \code
  ///   evalItervarRange(rmap);
  /// \encode
  ///
  /// \param r the map which records the newest values
  /// of IterVars.
  ///
  /// \return None.
  void evalItervarRange(const Rmap& r);

 private:
  /// minimal boundary.
  ir::ExprPtr min_value;

  /// maximal boundary.
  ir::ExprPtr max_value;
};

IntSet merge(const IntSet& ls, const IntSet& rs);
IntSet merge(std::vector<IntSetPtr>* IntSet_vector);

IntSet ceil(const IntSet& s);

void evalAllItervarRange(Rmap& r);  // NOLINT

#endif  // ELENA_INCLUDE_IR_INTSET_H_
