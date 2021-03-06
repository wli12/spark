/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.typesafe.tools.mima.core.{MissingTypesProblem, MissingClassProblem, ProblemFilters}
import com.typesafe.tools.mima.core.ProblemFilters._
import com.typesafe.tools.mima.plugin.MimaKeys.{binaryIssueFilters, previousArtifact}
import com.typesafe.tools.mima.plugin.MimaPlugin.mimaDefaultSettings
import sbt._

object MimaBuild {
  // Exclude a single class and its corresponding object
  def excludeClass(className: String) = {
    Seq(
      excludePackage(className),
      ProblemFilters.exclude[MissingClassProblem](className),
      ProblemFilters.exclude[MissingTypesProblem](className),
      excludePackage(className + "$"),
      ProblemFilters.exclude[MissingClassProblem](className + "$"),
      ProblemFilters.exclude[MissingTypesProblem](className + "$")
    )
  }
  // Exclude a Spark class, that is in the package org.apache.spark
  def excludeSparkClass(className: String) = {
    excludeClass("org.apache.spark." + className)
  }

  // Exclude a Spark package, that is in the package org.apache.spark
  def excludeSparkPackage(packageName: String) = {
    excludePackage("org.apache.spark." + packageName)
  }

  def ignoredABIProblems(base: File) = {

    // Excludes placed here will be used for all Spark versions
    val defaultExcludes = Seq()

    // Read package-private excludes from file
    val excludeFilePath = (base.getAbsolutePath + "/.generated-mima-excludes")
    val excludeFile = file(excludeFilePath)
    val ignoredClasses: Seq[String] =
      if (!excludeFile.exists()) {
        Seq()
      } else {
        IO.read(excludeFile).split("\n")
      }



    val externalExcludeFileClasses = ignoredClasses.flatMap(excludeClass)

    defaultExcludes ++ externalExcludeFileClasses ++ MimaExcludes.excludes
  }

  def mimaSettings(sparkHome: File) = mimaDefaultSettings ++ Seq(
    previousArtifact := None,
    binaryIssueFilters ++= ignoredABIProblems(sparkHome)
  )
}
