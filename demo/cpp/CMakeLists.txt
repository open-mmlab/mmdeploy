set(TASK_LIST ${MMDEPLOY_TASKS})

if ("text_detector" IN_LIST TASK_LIST)
    list(REMOVE_ITEM TASK_LIST "text_detector" "text_recognizer")
    add_subdirectory(ocr)
endif()

foreach(TASK ${TASK_LIST})
    add_subdirectory(${TASK})
endforeach()

if ("detector" IN_LIST TASK_LIST AND "classifier" IN_LIST TASK_LIST)
    add_subdirectory(det_cls)
endif()

if ("detector" IN_LIST TASK_LIST AND "pose_detector" IN_LIST TASK_LIST)
    add_subdirectory(det_pose)
    add_subdirectory(pose_tracker)
endif()
