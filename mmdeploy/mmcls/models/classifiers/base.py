from mmdeploy.utils import FUNCTION_REWRITERS


@FUNCTION_REWRITERS.register_rewriter(
    'mmcls.models.classifiers.ImageClassifier.forward', backend='default')
@FUNCTION_REWRITERS.register_rewriter(
    'mmcls.models.classifiers.BaseClassifier.forward', backend='default')
def BaseClassifier_forward_default_wrapper(rewriter, self, img, *args,
                                           **kwargs):
    return self.simple_test(img, {})
